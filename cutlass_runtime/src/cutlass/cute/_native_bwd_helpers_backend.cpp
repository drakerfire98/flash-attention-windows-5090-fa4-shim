#include <torch/extension.h>

namespace {

void flash_attn_backward_preprocess_zero(
    const c10::optional<torch::Tensor>& dpsum_opt,
    const c10::optional<torch::Tensor>& lse_log2_opt,
    const c10::optional<torch::Tensor>& dq_accum_opt) {
  if (dpsum_opt.has_value() && dpsum_opt.value().defined()) {
    dpsum_opt.value().zero_();
  }
  if (lse_log2_opt.has_value() && lse_log2_opt.value().defined()) {
    lse_log2_opt.value().zero_();
  }
  if (dq_accum_opt.has_value() && dq_accum_opt.value().defined()) {
    dq_accum_opt.value().zero_();
  }
}

void flash_attn_backward_postprocess_copy(
    const torch::Tensor& accum,
    const torch::Tensor& output) {
  TORCH_CHECK(output.defined(), "output must be defined");
  if (output.sizes().equals(accum.sizes())) {
    output.copy_(accum);
    return;
  }
  if (output.dim() == 3 && accum.dim() == 3) {
    output.copy_(accum.permute({0, 2, 1}).contiguous());
    return;
  }
  if (output.dim() == 2 && accum.dim() == 2) {
    output.copy_(accum.transpose(-1, -2).contiguous());
    return;
  }
  TORCH_CHECK(
      false,
      "Cannot copy accumulated tensor with shape ",
      accum.sizes(),
      " into output shape ",
      output.sizes());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_attn_backward_preprocess_zero",
      &flash_attn_backward_preprocess_zero,
      "Compiled FA4 backward preprocess zero helper for Windows");
  m.def(
      "flash_attn_backward_postprocess_copy",
      &flash_attn_backward_postprocess_copy,
      "Compiled FA4 backward postprocess copy helper for Windows");
}
