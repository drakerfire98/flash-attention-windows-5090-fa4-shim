#include <torch/extension.h>

#include <limits>
#include <vector>

namespace {

torch::Tensor build_causal_mask(
    int64_t seqlen_q,
    int64_t seqlen_k,
    const torch::Device& device) {
  auto options = torch::TensorOptions().device(device).dtype(torch::kLong);
  auto q_idx = torch::arange(seqlen_q, options).unsqueeze(1);
  auto k_idx = torch::arange(seqlen_k, options).unsqueeze(0);
  auto relative = k_idx - (q_idx + seqlen_k - seqlen_q);
  return relative <= 0;
}

std::vector<torch::Tensor> flash_attn_dense_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double softmax_scale,
    bool causal) {
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
  if (causal) {
    auto keep_mask = build_causal_mask(q.size(1), k.size(1), scores.device())
                         .view({1, 1, q.size(1), k.size(1)});
    scores = scores.masked_fill(~keep_mask, -std::numeric_limits<float>::infinity());
  }

  auto valid = ~torch::isneginf(scores);
  auto all_masked = ~valid.any(-1, true);
  auto row_max = std::get<0>(scores.max(-1, true));
  auto safe_row_max = torch::where(all_masked, torch::zeros_like(row_max), row_max);
  auto exp_scores = torch::where(
      valid,
      torch::exp(scores - safe_row_max),
      torch::zeros_like(scores));
  auto normalizer = exp_scores.sum(-1, true);
  auto probs = torch::where(
      normalizer > 0,
      exp_scores / normalizer,
      torch::zeros_like(exp_scores));
  auto lse = torch::where(
                 normalizer > 0,
                 torch::log(normalizer) + safe_row_max,
                 torch::full_like(
                     safe_row_max,
                     -std::numeric_limits<float>::infinity()))
                 .squeeze(-1)
                 .permute({0, 2, 1})
                 .contiguous();

  auto out = torch::matmul(probs, v_t).to(q.dtype()).permute({0, 2, 1, 3}).contiguous();
  return {out, lse};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_attn_dense_forward",
      &flash_attn_dense_forward,
      "Compiled dense FA4 forward backend for Windows");
}
