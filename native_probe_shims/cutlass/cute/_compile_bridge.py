"""Bridge selected FA4 CuTe compile calls onto the validated Windows shim path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch

from cutlass.cutlass_dsl import JitCompiledFunction
from _probe_helpers import ProbePlaceholder


_SHIM_MODULE: ModuleType | None = None
_DLSE_BY_DQACCUM: dict[int, torch.Tensor | None] = {}
_POSTPROCESS_RESULTS: dict[int, torch.Tensor] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_windows_shim_module() -> ModuleType:
    global _SHIM_MODULE
    if _SHIM_MODULE is not None:
        return _SHIM_MODULE

    shim_init = _repo_root() / "shims" / "flash_attn" / "cute" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "native_probe_windows_flash_attn_cute_shim",
        shim_init,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Windows shim module from {shim_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SHIM_MODULE = module
    return module


def _copy_tensor_like(dst: torch.Tensor | None, src: torch.Tensor | None) -> None:
    if dst is None or src is None:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    if dst.ndim == 3 and src.ndim == 3:
        dst.copy_(src.permute(0, 2, 1).contiguous())
        return
    if dst.ndim == 2 and src.ndim == 2:
        dst.copy_(src.transpose(-1, -2).contiguous())
        return
    raise ValueError(f"Cannot copy shim tensor with shape {src.shape} into target shape {dst.shape}")


def _convert_tensor_like(src: torch.Tensor | None, like: torch.Tensor | None) -> torch.Tensor | None:
    if src is None or like is None:
        return None
    if src.shape == like.shape:
        return src
    if src.ndim == 3 and like.ndim == 3:
        return src.permute(0, 2, 1).contiguous()
    if src.ndim == 2 and like.ndim == 2:
        return src.transpose(-1, -2).contiguous()
    raise ValueError(f"Cannot convert tensor with shape {src.shape} to match {like.shape}")


def _window_tuple(window_left: int | None, window_right: int | None) -> tuple[int | None, int | None]:
    return (window_left, window_right)


def _extract_softcap_from_generated_score_mod(score_mod: Any) -> float | None:
    """Best-effort detection of FA4's internal softcap wrapper.

    In the upstream forward path, a user-provided ``softcap`` is rewritten into
    an internal ``score_mod`` closure via ``utils.create_softcap_scoremod``.
    That generated callable is CuTe-flavored Python and is not executable in
    our Windows bridge path, so we recover the original numeric softcap value
    from the closure and route it through the stable shim's native ``softcap``
    support instead.
    """

    if not callable(score_mod):
        return None
    if getattr(score_mod, "__name__", "") != "scoremod_premask_fn":
        return None
    for cell in getattr(score_mod, "__closure__", ()) or ():
        value = getattr(cell, "cell_contents", None)
        if isinstance(value, (int, float)) and value > 0:
            return 1.0 / float(value)
    return None


class NativeProbeForwardBridge(JitCompiledFunction):
    def __init__(self, kernel: Any):
        self.kernel = kernel
        self.kernel_name = type(kernel).__name__

    def __repr__(self) -> str:
        return f"<NativeProbeForwardBridge {self.kernel_name}>"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor | None,
        softmax_scale: float | None,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        seqused_q: torch.Tensor | None,
        seqused_k: torch.Tensor | None,
        page_table: torch.Tensor | None,
        window_size_left: int | None,
        window_size_right: int | None,
        learnable_sink: torch.Tensor | None,
        block_sparse_tensors: Any,
        aux_tensors: list[torch.Tensor] | None,
    ) -> None:
        if block_sparse_tensors is not None:
            raise NotImplementedError("Block-sparse native probe bridge is not implemented yet")

        shim = _load_windows_shim_module()
        common = dict(
            softmax_scale=softmax_scale,
            causal=bool(getattr(self.kernel, "is_causal", False)),
            window_size=_window_tuple(window_size_left, window_size_right),
            learnable_sink=learnable_sink,
            return_lse=lse is not None,
        )

        is_varlen = any(t is not None for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table))
        score_mod = getattr(self.kernel, "score_mod", None)
        mask_mod = getattr(self.kernel, "mask_mod", None)
        softcap = _extract_softcap_from_generated_score_mod(score_mod)
        if softcap is not None:
            score_mod = None

        if is_varlen:
            shim_out, shim_lse = shim.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
                softcap=softcap or 0.0,
                **common,
            )
        elif score_mod is not None:
            shim_out, shim_lse = shim._attention_forward_dense(
                q,
                k,
                v,
                score_mod=score_mod,
                mask_mod=mask_mod,
                aux_tensors=aux_tensors,
                batch_start_index=0,
                offset_q=0,
                offset_k=0,
                softcap=softcap or 0.0,
                **common,
            )
        else:
            shim_out, shim_lse = shim.flash_attn_func(
                q,
                k,
                v,
                softcap=softcap or 0.0,
                mask_mod=mask_mod,
                **common,
            )

        _copy_tensor_like(out, shim_out)
        _copy_tensor_like(lse, shim_lse)


class NativeProbeBackwardPreprocessBridge(JitCompiledFunction):
    def __repr__(self) -> str:
        return "<NativeProbeBackwardPreprocessBridge>"

    def __call__(
        self,
        out: torch.Tensor,
        dout: torch.Tensor,
        dpsum: torch.Tensor,
        lse: torch.Tensor | None,
        lse_log2: torch.Tensor | None,
        dq_accum: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor | None,
        seqused_q: torch.Tensor | None,
        dlse: torch.Tensor | None,
    ) -> None:
        del out, dout, lse, cu_seqlens_q, seqused_q
        if dpsum is not None:
            dpsum.zero_()
        if lse_log2 is not None:
            lse_log2.zero_()
        if dq_accum is not None:
            dq_accum.zero_()
            _DLSE_BY_DQACCUM[dq_accum.data_ptr()] = dlse.detach().clone() if dlse is not None else None


class NativeProbeBackwardPostprocessBridge(JitCompiledFunction):
    def __repr__(self) -> str:
        return "<NativeProbeBackwardPostprocessBridge>"

    def __call__(
        self,
        accum: torch.Tensor,
        output: torch.Tensor,
        scale: float | torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        seqused: torch.Tensor | None,
    ) -> None:
        del scale, cu_seqlens, seqused
        result = _POSTPROCESS_RESULTS.pop(accum.data_ptr(), None)
        if result is None:
            raise NotImplementedError(
                "Native probe postprocess bridge has no staged gradient result for this accumulator"
            )
        _copy_tensor_like(output, result)


class NativeProbeBackwardBridge(JitCompiledFunction):
    def __init__(self, kernel: Any):
        self.kernel = kernel
        self.kernel_name = type(kernel).__name__

    def __repr__(self) -> str:
        return f"<NativeProbeBackwardBridge {self.kernel_name}>"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        lse_log2: torch.Tensor,
        dpsum: torch.Tensor,
        dq_accum: torch.Tensor,
        dk_or_accum: torch.Tensor,
        dv_or_accum: torch.Tensor,
        softmax_scale: float | None,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        seqused_q: torch.Tensor | None,
        seqused_k: torch.Tensor | None,
        softcap: Any,
        window_size_left: int | None,
        window_size_right: int | None,
        dq_semaphore: Any,
        dk_semaphore: Any,
        dv_semaphore: Any,
        aux_tensors: list[torch.Tensor] | None,
        block_sparse_tensors: Any,
    ) -> None:
        del lse_log2, dpsum, dq_semaphore, dk_semaphore, dv_semaphore
        if block_sparse_tensors is not None:
            raise NotImplementedError("Block-sparse native probe backward bridge is not implemented yet")

        shim = _load_windows_shim_module()
        dlse = _DLSE_BY_DQACCUM.pop(dq_accum.data_ptr(), None)
        softcap_value = 0.0 if softcap is None else float(softcap)

        with torch.enable_grad():
            q_req = q.detach().clone().requires_grad_(True)
            k_req = k.detach().clone().requires_grad_(True)
            v_req = v.detach().clone().requires_grad_(True)

            common = dict(
                softmax_scale=softmax_scale,
                causal=bool(getattr(self.kernel, "is_causal", False)),
                window_size=_window_tuple(window_size_left, window_size_right),
                return_lse=dlse is not None,
            )

            is_varlen = any(t is not None for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k))
            score_mod = getattr(self.kernel, "score_mod", None)
            mask_mod = getattr(self.kernel, "mask_mod", None)

            if is_varlen:
                shim_out, shim_lse = shim.flash_attn_varlen_func(
                    q_req,
                    k_req,
                    v_req,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    page_table=None,
                    score_mod=score_mod,
                    aux_tensors=aux_tensors,
                    learnable_sink=None,
                    softcap=softcap_value,
                    **common,
                )
            elif score_mod is not None:
                shim_out, shim_lse = shim._attention_forward_dense(
                    q_req,
                    k_req,
                    v_req,
                    learnable_sink=None,
                    softcap=softcap_value,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    aux_tensors=aux_tensors,
                    batch_start_index=0,
                    offset_q=0,
                    offset_k=0,
                    **common,
                )
            else:
                shim_out, shim_lse = shim.flash_attn_func(
                    q_req,
                    k_req,
                    v_req,
                    learnable_sink=None,
                    softcap=softcap_value,
                    mask_mod=mask_mod,
                    **common,
                )

            grad_outputs: list[torch.Tensor] = [dout]
            outputs: list[torch.Tensor] = [shim_out]
            if dlse is not None and shim_lse is not None:
                outputs.append(shim_lse)
                grad_outputs.append(_convert_tensor_like(dlse, shim_lse))

            dq_grad, dk_grad, dv_grad = torch.autograd.grad(
                outputs=tuple(outputs),
                inputs=(q_req, k_req, v_req),
                grad_outputs=tuple(grad_outputs),
                retain_graph=False,
                allow_unused=False,
            )

        _POSTPROCESS_RESULTS[dq_accum.data_ptr()] = dq_grad.detach()

        qhead_per_kvhead = int(getattr(self.kernel, "qhead_per_kvhead", 1))
        if qhead_per_kvhead > 1:
            _POSTPROCESS_RESULTS[dk_or_accum.data_ptr()] = dk_grad.detach()
            _POSTPROCESS_RESULTS[dv_or_accum.data_ptr()] = dv_grad.detach()
        else:
            _copy_tensor_like(dk_or_accum, dk_grad.detach())
            _copy_tensor_like(dv_or_accum, dv_grad.detach())


class NativeProbeUnsupportedBridge(JitCompiledFunction):
    def __init__(self, op: Any):
        self.op = op
        self.op_name = type(op).__name__

    def __repr__(self) -> str:
        return f"<NativeProbeUnsupportedBridge {self.op_name}>"

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            f"Native probe compile bridge does not yet implement {self.op_name}"
        )


def compile_dispatch(op: Any, *args, **kwargs):
    del args, kwargs
    op_name = type(op).__name__
    if op_name.startswith("FlashAttentionForwardSm"):
        return NativeProbeForwardBridge(op)
    if op_name == "FlashAttentionBackwardPreprocess":
        return NativeProbeBackwardPreprocessBridge()
    if op_name == "FlashAttentionBackwardPostprocess":
        return NativeProbeBackwardPostprocessBridge()
    if op_name.startswith("FlashAttentionBackwardSm"):
        return NativeProbeBackwardBridge(op)
    return NativeProbeUnsupportedBridge(op)


def compile_repr() -> ProbePlaceholder:
    return ProbePlaceholder("cutlass.cute.compile")
