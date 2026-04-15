#include <torch/extension.h>
#include <torch/csrc/autograd/autograd.h>

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

std::vector<torch::Tensor> dense_forward_outputs(
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
  TORCH_CHECK(q.dim() == 4, "q must be shaped (batch, seqlen_q, heads, dim)");
  TORCH_CHECK(k.dim() == 4, "k must be shaped (batch, seqlen_k, heads, dim)");
  TORCH_CHECK(v.dim() == 4, "v must be shaped (batch, seqlen_k, heads, dim_v)");
  TORCH_CHECK(q.device() == k.device(), "q and k must be on the same device");
  TORCH_CHECK(q.device() == v.device(), "q and v must be on the same device");
  TORCH_CHECK(q.size(0) == k.size(0), "q and k batch sizes must match");
  TORCH_CHECK(q.size(0) == v.size(0), "q and v batch sizes must match");
  TORCH_CHECK(k.size(1) == v.size(1), "k and v seqlen_k must match");
  TORCH_CHECK(k.size(3) == q.size(3), "q and k head dims must match");
  TORCH_CHECK(k.size(2) == v.size(2), "k and v head counts must match");

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
        "extra_keep_mask must be shaped (batch, heads, seqlen_q, seqlen_k)");
    TORCH_CHECK(
        extra_keep_mask.size(0) == scores.size(0) &&
            extra_keep_mask.size(1) == scores.size(1) &&
            extra_keep_mask.size(2) == scores.size(2) &&
            extra_keep_mask.size(3) == scores.size(3),
        "extra_keep_mask shape must match the expanded attention score shape");
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

std::vector<torch::Tensor> flash_attn_dense_backward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& dout,
    const c10::optional<torch::Tensor>& dlse_opt,
    double softmax_scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap,
    const c10::optional<torch::Tensor>& learnable_sink_opt,
    const c10::optional<torch::Tensor>& extra_keep_mask_opt) {
  TORCH_CHECK(dout.defined(), "dout must be defined");
  TORCH_CHECK(dout.dim() == 4, "dout must be shaped (batch, seqlen_q, heads, dim_v)");
  TORCH_CHECK(dout.size(0) == q.size(0), "dout batch size must match q");
  TORCH_CHECK(dout.size(1) == q.size(1), "dout seqlen_q must match q");
  TORCH_CHECK(dout.size(2) == q.size(2), "dout head count must match q");
  TORCH_CHECK(dout.size(3) == v.size(3), "dout head dim must match v");
  if (dlse_opt.has_value() && dlse_opt.value().defined()) {
    const auto& dlse = dlse_opt.value();
    TORCH_CHECK(dlse.dim() == 3, "dlse must be shaped (batch, seqlen_q, heads)");
    TORCH_CHECK(dlse.size(0) == q.size(0), "dlse batch size must match q");
    TORCH_CHECK(dlse.size(1) == q.size(1), "dlse seqlen_q must match q");
    TORCH_CHECK(dlse.size(2) == q.size(2), "dlse head count must match q");
  }

  torch::autograd::AutoGradMode enable_grad(true);
  auto q_req = q.detach().clone().set_requires_grad(true);
  auto k_req = k.detach().clone().set_requires_grad(true);
  auto v_req = v.detach().clone().set_requires_grad(true);

  auto forward_outputs = dense_forward_outputs(
      q_req,
      k_req,
      v_req,
      softmax_scale,
      causal,
      window_left,
      window_right,
      softcap,
      learnable_sink_opt,
      extra_keep_mask_opt);

  std::vector<torch::Tensor> outputs = {forward_outputs[0]};
  std::vector<torch::Tensor> grad_outputs = {dout};
  if (dlse_opt.has_value() && dlse_opt.value().defined()) {
    outputs.push_back(forward_outputs[1]);
    grad_outputs.push_back(dlse_opt.value());
  }

  auto grads = torch::autograd::grad(
      outputs,
      {q_req, k_req, v_req},
      grad_outputs,
      /*retain_graph=*/false,
      /*create_graph=*/false,
      /*allow_unused=*/false);
  return grads;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_attn_dense_backward",
      [](const torch::Tensor& q,
         const torch::Tensor& k,
         const torch::Tensor& v,
         const torch::Tensor& dout,
         const c10::optional<torch::Tensor>& dlse_opt,
         double softmax_scale,
         bool causal,
         int64_t window_left,
         int64_t window_right,
         double softcap,
         const c10::optional<torch::Tensor>& learnable_sink_opt,
         const c10::optional<torch::Tensor>& extra_keep_mask_opt) {
        pybind11::gil_scoped_release no_gil;
        return flash_attn_dense_backward(
            q,
            k,
            v,
            dout,
            dlse_opt,
            softmax_scale,
            causal,
            window_left,
            window_right,
            softcap,
            learnable_sink_opt,
            extra_keep_mask_opt);
      },
      "Compiled dense FA4 backward backend for Windows");
}
