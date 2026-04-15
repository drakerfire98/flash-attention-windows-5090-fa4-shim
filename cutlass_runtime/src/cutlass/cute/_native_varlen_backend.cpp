#include <torch/extension.h>

#include <limits>
#include <vector>

namespace {

torch::Tensor build_attention_mask(
    int64_t seqlen_q,
    int64_t seqlen_k,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    const torch::Device& device) {
  auto options = torch::TensorOptions().device(device).dtype(torch::kLong);
  auto q_idx = torch::arange(seqlen_q, options).unsqueeze(1);
  auto k_idx = torch::arange(seqlen_k, options).unsqueeze(0);
  auto relative = k_idx - (q_idx + seqlen_k - seqlen_q);
  auto allowed =
      torch::ones({seqlen_q, seqlen_k}, torch::TensorOptions().device(device).dtype(torch::kBool));
  if (causal) {
    allowed = allowed & (relative <= 0);
  }
  if (window_left >= 0) {
    allowed = allowed & (relative >= -window_left);
  }
  if (window_right >= 0) {
    allowed = allowed & (relative <= window_right);
  }
  return allowed;
}

std::vector<torch::Tensor> dense_forward_chunk(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double softmax_scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap,
    const c10::optional<torch::Tensor>& learnable_sink_opt,
    const c10::optional<torch::Tensor>& extra_keep_mask_opt) {
  TORCH_CHECK(q.dim() == 4, "q chunk must be shaped (batch, seqlen_q, heads, dim)");
  TORCH_CHECK(k.dim() == 4, "k chunk must be shaped (batch, seqlen_k, heads, dim)");
  TORCH_CHECK(v.dim() == 4, "v chunk must be shaped (batch, seqlen_k, heads, dim_v)");

  auto q_t = q.permute({0, 2, 1, 3}).contiguous().to(torch::kFloat);
  auto k_t = k.permute({0, 2, 1, 3}).contiguous().to(torch::kFloat);
  auto v_t = v.permute({0, 2, 1, 3}).contiguous().to(torch::kFloat);

  const auto q_heads = q_t.size(1);
  const auto kv_heads = k_t.size(1);
  if (q_heads != kv_heads) {
    TORCH_CHECK(
        q_heads % kv_heads == 0,
        "q heads (",
        q_heads,
        ") must equal or be divisible by kv heads (",
        kv_heads,
        ")");
    const auto repeat = q_heads / kv_heads;
    k_t = k_t.repeat_interleave(repeat, 1);
    v_t = v_t.repeat_interleave(repeat, 1);
  }

  auto scores = torch::matmul(q_t, k_t.transpose(-1, -2)) * softmax_scale;
  if (causal || window_left >= 0 || window_right >= 0) {
    auto keep_mask = build_attention_mask(
                         q.size(1),
                         k.size(1),
                         causal,
                         window_left,
                         window_right,
                         scores.device())
                         .view({1, 1, q.size(1), k.size(1)});
    scores = scores.masked_fill(~keep_mask, -std::numeric_limits<float>::infinity());
  }
  if (extra_keep_mask_opt.has_value() && extra_keep_mask_opt.value().defined()) {
    auto extra_keep_mask = extra_keep_mask_opt.value().to(scores.device(), torch::kBool);
    TORCH_CHECK(
        extra_keep_mask.dim() == 4,
        "extra_keep_mask chunk must be shaped (batch, heads, seqlen_q, seqlen_k)");
    TORCH_CHECK(
        extra_keep_mask.size(0) == scores.size(0) &&
            extra_keep_mask.size(1) == scores.size(1) &&
            extra_keep_mask.size(2) == scores.size(2) &&
            extra_keep_mask.size(3) == scores.size(3),
        "extra_keep_mask chunk shape must match the expanded attention score shape");
    scores = scores.masked_fill(~extra_keep_mask, -std::numeric_limits<float>::infinity());
  }
  if (softcap > 0.0) {
    scores = torch::tanh(scores / softcap) * softcap;
  }

  torch::Tensor probs;
  torch::Tensor lse;
  if (learnable_sink_opt.has_value() && learnable_sink_opt.value().defined()) {
    auto sink = learnable_sink_opt.value().to(scores.device(), torch::kFloat);
    TORCH_CHECK(sink.dim() == 1, "learnable_sink must be 1D");
    TORCH_CHECK(
        sink.size(0) == scores.size(1),
        "learnable_sink length must match the query head count");
    sink = sink.view({1, scores.size(1), 1, 1});

    auto logits_max = std::get<0>(scores.max(-1, true));
    auto logits_or_sink_max = torch::maximum(logits_max, sink);
    auto unnormalized_scores = torch::exp(scores - logits_or_sink_max);
    auto normalizer =
        unnormalized_scores.sum(-1, true) + torch::exp(sink - logits_or_sink_max);
    probs = unnormalized_scores / normalizer;
    lse = (torch::log(normalizer) + logits_or_sink_max)
              .squeeze(-1)
              .permute({0, 2, 1})
              .contiguous();
  } else {
    auto valid = ~torch::isneginf(scores);
    auto all_masked = ~valid.any(-1, true);
    auto row_max = std::get<0>(scores.max(-1, true));
    auto safe_row_max = torch::where(all_masked, torch::zeros_like(row_max), row_max);
    auto exp_scores = torch::where(
        valid,
        torch::exp(scores - safe_row_max),
        torch::zeros_like(scores));
    auto normalizer = exp_scores.sum(-1, true);
    probs = torch::where(
        normalizer > 0,
        exp_scores / normalizer,
        torch::zeros_like(exp_scores));
    lse = torch::where(
              normalizer > 0,
              torch::log(normalizer) + safe_row_max,
              torch::full_like(
                  safe_row_max,
                  -std::numeric_limits<float>::infinity()))
              .squeeze(-1)
              .permute({0, 2, 1})
              .contiguous();
  }

  auto out = torch::matmul(probs, v_t).to(q.dtype()).permute({0, 2, 1, 3}).contiguous();
  return {out, lse};
}

std::vector<torch::Tensor> flash_attn_varlen_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& q_starts,
    const torch::Tensor& q_used,
    bool q_packed,
    const torch::Tensor& k_starts,
    const torch::Tensor& k_used,
    bool k_packed,
    double softmax_scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap,
    const c10::optional<torch::Tensor>& learnable_sink_opt,
    const c10::optional<torch::Tensor>& extra_keep_mask_opt) {
  TORCH_CHECK(q.device() == k.device(), "q and k must be on the same device");
  TORCH_CHECK(q.device() == v.device(), "q and v must be on the same device");
  TORCH_CHECK(
      q.dtype() == k.dtype() && q.dtype() == v.dtype(),
      "q, k, and v must share the same dtype");

  auto q_starts_cpu = q_starts.to(torch::kCPU, torch::kLong).contiguous();
  auto q_used_cpu = q_used.to(torch::kCPU, torch::kLong).contiguous();
  auto k_starts_cpu = k_starts.to(torch::kCPU, torch::kLong).contiguous();
  auto k_used_cpu = k_used.to(torch::kCPU, torch::kLong).contiguous();

  TORCH_CHECK(q_starts_cpu.dim() == 1, "q_starts must be 1D");
  TORCH_CHECK(q_used_cpu.dim() == 1, "q_used must be 1D");
  TORCH_CHECK(k_starts_cpu.dim() == 1, "k_starts must be 1D");
  TORCH_CHECK(k_used_cpu.dim() == 1, "k_used must be 1D");
  TORCH_CHECK(
      q_starts_cpu.numel() == q_used_cpu.numel(),
      "q_starts and q_used must have the same length");
  TORCH_CHECK(
      k_starts_cpu.numel() == k_used_cpu.numel(),
      "k_starts and k_used must have the same length");
  TORCH_CHECK(
      q_used_cpu.numel() == k_used_cpu.numel(),
      "q and k layouts must describe the same batch size");

  const auto batch = q_used_cpu.numel();
  const auto q_heads = q_packed ? q.size(1) : q.size(2);
  const auto v_dim = v.size(-1);
  if (extra_keep_mask_opt.has_value() && extra_keep_mask_opt.value().defined()) {
    auto extra_keep_mask = extra_keep_mask_opt.value();
    TORCH_CHECK(
        extra_keep_mask.dim() == 4,
        "extra_keep_mask must be shaped (batch, heads, max_q, max_k)");
    TORCH_CHECK(
        extra_keep_mask.size(0) == batch && extra_keep_mask.size(1) == q_heads,
        "extra_keep_mask batch/head dims must match the q layout");
  }

  if (q_packed) {
    TORCH_CHECK(q.dim() == 3, "packed q must be shaped (total_q, heads, dim)");
  } else {
    TORCH_CHECK(q.dim() == 4, "padded q must be shaped (batch, seqlen_q, heads, dim)");
    TORCH_CHECK(q.size(0) == batch, "padded q batch size must match q layout");
  }
  if (k_packed) {
    TORCH_CHECK(k.dim() == 3, "packed k must be shaped (total_k, heads, dim)");
    TORCH_CHECK(v.dim() == 3, "packed v must be shaped (total_k, heads, dim_v)");
  } else {
    TORCH_CHECK(k.dim() == 4, "padded k must be shaped (batch, seqlen_k, heads, dim)");
    TORCH_CHECK(v.dim() == 4, "padded v must be shaped (batch, seqlen_k, heads, dim_v)");
    TORCH_CHECK(k.size(0) == batch, "padded k batch size must match k layout");
    TORCH_CHECK(v.size(0) == batch, "padded v batch size must match k layout");
  }

  std::vector<torch::Tensor> out_chunks;
  std::vector<torch::Tensor> lse_chunks;

  torch::Tensor out;
  torch::Tensor lse;
  if (!q_packed) {
    out = torch::zeros({batch, q.size(1), q_heads, v_dim}, q.options());
    lse = torch::full(
        {batch, q.size(1), q_heads},
        -std::numeric_limits<float>::infinity(),
        q.options().dtype(torch::kFloat));
  }

  const auto q_starts_acc = q_starts_cpu.accessor<int64_t, 1>();
  const auto q_used_acc = q_used_cpu.accessor<int64_t, 1>();
  const auto k_starts_acc = k_starts_cpu.accessor<int64_t, 1>();
  const auto k_used_acc = k_used_cpu.accessor<int64_t, 1>();

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const auto q_start = q_starts_acc[batch_idx];
    const auto q_len = q_used_acc[batch_idx];
    const auto k_start = k_starts_acc[batch_idx];
    const auto k_len = k_used_acc[batch_idx];

    TORCH_CHECK(q_len >= 0, "q_used must be non-negative");
    TORCH_CHECK(k_len >= 0, "k_used must be non-negative");
    if (q_packed) {
      TORCH_CHECK(q_start >= 0, "q_starts must be non-negative");
      TORCH_CHECK(q_start + q_len <= q.size(0), "packed q layout exceeds q length");
    } else {
      TORCH_CHECK(q_len <= q.size(1), "padded q used length exceeds padded q length");
    }
    if (k_packed) {
      TORCH_CHECK(k_start >= 0, "k_starts must be non-negative");
      TORCH_CHECK(k_start + k_len <= k.size(0), "packed k layout exceeds k length");
      TORCH_CHECK(k_start + k_len <= v.size(0), "packed v layout exceeds v length");
    } else {
      TORCH_CHECK(k_len <= k.size(1), "padded k used length exceeds padded k length");
      TORCH_CHECK(k_len <= v.size(1), "padded v used length exceeds padded v length");
    }

    auto q_chunk = q_packed ? q.narrow(0, q_start, q_len).unsqueeze(0)
                            : q.select(0, batch_idx).narrow(0, 0, q_len).unsqueeze(0);
    auto k_chunk = k_packed ? k.narrow(0, k_start, k_len).unsqueeze(0)
                            : k.select(0, batch_idx).narrow(0, 0, k_len).unsqueeze(0);
    auto v_chunk = k_packed ? v.narrow(0, k_start, k_len).unsqueeze(0)
                            : v.select(0, batch_idx).narrow(0, 0, k_len).unsqueeze(0);

    torch::Tensor out_chunk;
    torch::Tensor lse_chunk;
    c10::optional<torch::Tensor> extra_keep_mask_chunk = c10::nullopt;
    if (extra_keep_mask_opt.has_value() && extra_keep_mask_opt.value().defined()) {
        extra_keep_mask_chunk = extra_keep_mask_opt.value()
            .narrow(0, batch_idx, 1)
            .narrow(2, 0, q_len)
            .narrow(3, 0, k_len);
    }
    if (q_len == 0 || k_len == 0) {
      out_chunk = torch::zeros({1, q_len, q_heads, v_dim}, q.options());
      lse_chunk = torch::full(
          {1, q_len, q_heads},
          -std::numeric_limits<float>::infinity(),
          q.options().dtype(torch::kFloat));
    } else {
      auto chunk_result = dense_forward_chunk(
          q_chunk,
          k_chunk,
          v_chunk,
          softmax_scale,
          causal,
          window_left,
          window_right,
          softcap,
          learnable_sink_opt,
          extra_keep_mask_chunk);
      out_chunk = chunk_result[0];
      lse_chunk = chunk_result[1];
    }

    if (q_packed) {
      out_chunks.push_back(out_chunk.squeeze(0));
      lse_chunks.push_back(lse_chunk.squeeze(0));
    } else if (q_len > 0) {
      out.select(0, batch_idx).narrow(0, 0, q_len).copy_(out_chunk.squeeze(0));
      lse.select(0, batch_idx).narrow(0, 0, q_len).copy_(lse_chunk.squeeze(0));
    }
  }

  if (q_packed) {
    if (out_chunks.empty()) {
      out = q.new_empty({0, q_heads, v_dim});
      lse = torch::empty({0, q_heads}, q.options().dtype(torch::kFloat));
    } else {
      out = torch::cat(out_chunks, 0);
      lse = torch::cat(lse_chunks, 0);
    }
  }

  return {out, lse};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_attn_varlen_forward",
      &flash_attn_varlen_forward,
      "Compiled varlen FA4 forward backend for Windows");
}
