#include <torch/extension.h>

#include <limits>
#include <vector>

namespace {

std::vector<torch::Tensor> flash_attn_combine_forward(
    const torch::Tensor& out_partial,
    const torch::Tensor& lse_partial,
    const c10::optional<torch::Tensor>& split_valid_mask_opt) {
  TORCH_CHECK(
      out_partial.dim() == 4 || out_partial.dim() == 5,
      "out_partial must have 4 or 5 dimensions, got ",
      out_partial.dim());
  TORCH_CHECK(
      lse_partial.dim() + 1 == out_partial.dim(),
      "lse_partial must have exactly one fewer dimension than out_partial");
  TORCH_CHECK(
      out_partial.device() == lse_partial.device(),
      "out_partial and lse_partial must be on the same device");

  auto lse_float = lse_partial.to(torch::kFloat);
  auto valid = torch::isfinite(lse_float);
  if (split_valid_mask_opt.has_value() && split_valid_mask_opt.value().defined()) {
    auto split_valid_mask = split_valid_mask_opt.value().to(valid.device(), torch::kBool);
    TORCH_CHECK(
        split_valid_mask.sizes().equals(valid.sizes()),
        "split_valid_mask shape must match lse_partial shape");
    valid = valid & split_valid_mask;
  }

  auto any_valid = valid.any(0);
  auto lse_max = std::get<0>(lse_float.max(0));
  auto safe_lse_max = torch::where(any_valid, lse_max, torch::zeros_like(lse_max));
  auto weights = torch::where(
      valid,
      torch::exp(lse_float - safe_lse_max.unsqueeze(0)),
      torch::zeros_like(lse_float));
  auto denom = weights.sum(0);
  auto numerator = (out_partial.to(torch::kFloat) * weights.unsqueeze(-1)).sum(0);
  auto out = torch::where(
      denom.unsqueeze(-1) > 0,
      numerator / denom.unsqueeze(-1),
      torch::zeros_like(numerator));

  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  auto lse = torch::where(
      denom > 0,
      torch::log(denom) + safe_lse_max,
      torch::full_like(safe_lse_max, kNegInf));
  return {out, lse};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_attn_combine_forward",
      &flash_attn_combine_forward,
      "Compiled FA4 combine backend for Windows");
}
