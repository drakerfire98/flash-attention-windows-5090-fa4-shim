"""Runtime-local FA4 compile bridge for the Windows CuTe probe path."""

from __future__ import annotations

from typing import Any

import torch

from _probe_helpers import ProbePlaceholder
from cutlass.cutlass_dsl import JitCompiledFunction
from ._native_backend import flash_attn_combine_native, native_combine_backend_status
from ._native_bwd_helpers_backend import (
    flash_attn_backward_postprocess_copy_native,
    flash_attn_backward_preprocess_zero_native,
    native_bwd_helpers_backend_status,
)
from ._native_dense_backend import flash_attn_dense_forward_native, native_dense_backend_status
from ._native_varlen_backend import flash_attn_varlen_forward_native, native_varlen_backend_status
from ._runtime_local_core import (
    attention_forward_dense_local,
    attention_forward_varlen_local,
    fill_block_sparse_tensors,
    materialize_dense_keep_mask,
    materialize_varlen_keep_mask,
)


_DLSE_BY_DQACCUM: dict[int, torch.Tensor | None] = {}
_POSTPROCESS_RESULTS: dict[int, torch.Tensor] = {}
_FWD_METADATA_BY_TENSOR_PTR: dict[int, dict[str, Any]] = {}
_BWD_METADATA_BY_DQACCUM: dict[int, dict[str, Any]] = {}
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


def _can_use_native_dense_backend(
    *,
    is_varlen: bool,
    score_mod: Any,
    mask_mod: Any,
    softcap: float | None,
    learnable_sink: torch.Tensor | None,
    block_sparse_tensors: Any,
    window_size: tuple[int | None, int | None],
) -> bool:
    return (
        not is_varlen
        and score_mod is None
        and mask_mod is None
        and block_sparse_tensors is None
    )


def _can_use_native_varlen_backend(
    *,
    is_varlen: bool,
    score_mod: Any,
    mask_mod: Any,
    block_sparse_tensors: Any,
) -> bool:
    return (
        is_varlen
        and score_mod is None
        and mask_mod is None
        and block_sparse_tensors is None
    )


def _tensor_ptr(tensor: torch.Tensor | None) -> int | None:
    return None if tensor is None else int(tensor.data_ptr())


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _resolve_block_size(
    *,
    q_seqlen: int,
    tile_m: int,
    tile_n: int,
    block_sparse_tensors: Any,
    q_subtile_factor: int | None,
) -> tuple[int, int]:
    if q_subtile_factor is None:
        count_tensor = None
        if block_sparse_tensors is not None:
            count_tensor = block_sparse_tensors[0]
            if count_tensor is None and len(block_sparse_tensors) >= 3:
                count_tensor = block_sparse_tensors[2]
        if count_tensor is None or count_tensor.shape[-1] <= 0:
            q_subtile_factor = 1
        else:
            sparse_m_blocks = int(count_tensor.shape[-1])
            sparse_block_size_q = _ceil_div(_ceil_div(q_seqlen, sparse_m_blocks), tile_m) * tile_m
            q_subtile_factor = max(1, sparse_block_size_q // tile_m)
    return (tile_m * q_subtile_factor, tile_n)


def _stash_forward_metadata(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    *,
    softcap: float,
    score_mod: Any,
    mask_mod: Any,
    learnable_sink: torch.Tensor | None,
    aux_tensors: list[torch.Tensor] | None,
    page_table: torch.Tensor | None,
    block_sparse_tensors: Any,
    block_size: tuple[int, int] | None,
) -> None:
    keys = tuple(ptr for ptr in (_tensor_ptr(out), _tensor_ptr(lse)) if ptr is not None)
    if not keys:
        return
    metadata = {
        "cache_keys": keys,
        "softcap": float(softcap),
        "score_mod": score_mod,
        "mask_mod": mask_mod,
        "learnable_sink": learnable_sink,
        "aux_tensors": aux_tensors,
        "page_table": page_table,
        "block_sparse_tensors": block_sparse_tensors,
        "block_size": block_size,
    }
    for key in keys:
        _FWD_METADATA_BY_TENSOR_PTR[key] = metadata


def _pop_forward_metadata(out: torch.Tensor | None, lse: torch.Tensor | None) -> dict[str, Any] | None:
    metadata = None
    for key in (_tensor_ptr(out), _tensor_ptr(lse)):
        if key is None:
            continue
        metadata = _FWD_METADATA_BY_TENSOR_PTR.get(key)
        if metadata is not None:
            break
    if metadata is None:
        return None
    for key in metadata.get("cache_keys", ()):
        _FWD_METADATA_BY_TENSOR_PTR.pop(key, None)
    return {name: value for name, value in metadata.items() if name != "cache_keys"}


def _extract_softcap_score_mod_components(score_mod: Any) -> tuple[float | None, Any]:
    """Recover softcap and nested score_mod from FA4's closure wrapper when present."""

    if not callable(score_mod):
        return None, score_mod
    if getattr(score_mod, "__name__", "") != "scoremod_premask_fn":
        return None, score_mod
    softcap = None
    nested_score_mod = None
    for cell in getattr(score_mod, "__closure__", ()) or ():
        value = getattr(cell, "cell_contents", None)
        if isinstance(value, (int, float)) and value > 0:
            softcap = 1.0 / float(value)
        elif callable(value):
            nested_score_mod = value
    if softcap is None:
        return None, score_mod
    return softcap, nested_score_mod


def _resolve_bridge_block_size(
    kernel: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    block_sparse_tensors: Any,
) -> tuple[int, int] | None:
    if block_sparse_tensors is None:
        return None
    kernel_q_subtile_factor = getattr(kernel, "q_subtile_factor", None)
    q_subtile_factor = None if kernel_q_subtile_factor is None else int(kernel_q_subtile_factor or 1)
    tile_m = int(getattr(kernel, "tile_m", getattr(kernel, "m_block_size", q.shape[1])))
    tile_n = int(getattr(kernel, "tile_n", getattr(kernel, "n_block_size", k.shape[1])))
    q_seqlen = int(q.shape[1]) if q.ndim >= 2 else int(q.shape[0])
    return _resolve_block_size(
        q_seqlen=q_seqlen,
        tile_m=tile_m,
        tile_n=tile_n,
        block_sparse_tensors=block_sparse_tensors,
        q_subtile_factor=q_subtile_factor,
    )


def _run_forward_with_bridge_runtime(
    *,
    kernel: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    page_table: torch.Tensor | None,
    window_size: tuple[int | None, int | None],
    learnable_sink: torch.Tensor | None,
    softcap: float,
    score_mod: Any,
    mask_mod: Any,
    aux_tensors: list[torch.Tensor] | None,
    block_sparse_tensors: Any,
    block_size_override: tuple[int, int] | None = None,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int] | None]:
    is_varlen = any(t is not None for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table))
    causal = bool(getattr(kernel, "is_causal", False))
    block_size = block_size_override or _resolve_bridge_block_size(kernel, q, k, block_sparse_tensors)

    if score_mod is None and (mask_mod is not None or block_sparse_tensors is not None):
        if is_varlen:
            extra_keep_mask = materialize_varlen_keep_mask(
                q,
                k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                mask_mod=mask_mod,
                aux_tensors=aux_tensors,
                block_sparse_tensors=block_sparse_tensors,
                block_size=block_size,
            )
            try:
                return (
                    *flash_attn_varlen_forward_native(
                        q,
                        k,
                        v,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        page_table=page_table,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size,
                        learnable_sink=learnable_sink,
                        extra_keep_mask=extra_keep_mask,
                        softcap=softcap,
                        return_lse=return_lse,
                    ),
                    block_size,
                )
            except Exception:
                shim_out, shim_lse = attention_forward_varlen_local(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    page_table=page_table,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    softcap=softcap,
                    score_mod=None,
                    mask_mod=mask_mod,
                    aux_tensors=aux_tensors,
                    block_sparse_tensors=block_sparse_tensors,
                    block_size=block_size,
                    return_lse=return_lse,
                )
                return shim_out, shim_lse, block_size

        extra_keep_mask = materialize_dense_keep_mask(
            q,
            k,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            batch_start_index=0,
            offset_q=0,
            offset_k=0,
            block_sparse_tensors=block_sparse_tensors,
            block_size=block_size,
        )
        try:
            return (
                *flash_attn_dense_forward_native(
                    q,
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    extra_keep_mask=extra_keep_mask,
                    softcap=softcap,
                    return_lse=return_lse,
                ),
                block_size,
            )
        except Exception:
            shim_out, shim_lse = attention_forward_dense_local(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                softcap=softcap,
                score_mod=None,
                mask_mod=None,
                aux_tensors=aux_tensors,
                batch_start_index=0,
                offset_q=0,
                offset_k=0,
                extra_keep_mask=extra_keep_mask,
                return_lse=return_lse,
            )
            return shim_out, shim_lse, block_size

    if _can_use_native_varlen_backend(
        is_varlen=is_varlen,
        score_mod=score_mod,
        mask_mod=mask_mod,
        block_sparse_tensors=block_sparse_tensors,
    ):
        try:
            return (
                *flash_attn_varlen_forward_native(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_q=seqused_q,
                    seqused_k=seqused_k,
                    page_table=page_table,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    softcap=softcap,
                    return_lse=return_lse,
                ),
                block_size,
            )
        except Exception:
            shim_out, shim_lse = attention_forward_varlen_local(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                softcap=softcap,
                score_mod=None,
                mask_mod=None,
                aux_tensors=None,
                block_sparse_tensors=None,
                block_size=None,
                return_lse=return_lse,
            )
            return shim_out, shim_lse, block_size

    if _can_use_native_dense_backend(
        is_varlen=is_varlen,
        score_mod=score_mod,
        mask_mod=mask_mod,
        softcap=softcap,
        learnable_sink=learnable_sink,
        block_sparse_tensors=block_sparse_tensors,
        window_size=window_size,
    ):
        try:
            return (
                *flash_attn_dense_forward_native(
                    q,
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    softcap=softcap,
                    return_lse=return_lse,
                ),
                block_size,
            )
        except Exception:
            shim_out, shim_lse = attention_forward_dense_local(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                softcap=softcap,
                score_mod=None,
                mask_mod=None,
                aux_tensors=None,
                batch_start_index=0,
                offset_q=0,
                offset_k=0,
                extra_keep_mask=None,
                return_lse=return_lse,
            )
            return shim_out, shim_lse, block_size

    if is_varlen:
        shim_out, shim_lse = attention_forward_varlen_local(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=block_sparse_tensors,
            block_size=block_size,
            return_lse=return_lse,
        )
        return shim_out, shim_lse, block_size

    extra_keep_mask = None
    if block_sparse_tensors is not None:
        extra_keep_mask = materialize_dense_keep_mask(
            q,
            k,
            mask_mod=None,
            aux_tensors=None,
            batch_start_index=0,
            offset_q=0,
            offset_k=0,
            block_sparse_tensors=block_sparse_tensors,
            block_size=block_size,
        )
    shim_out, shim_lse = attention_forward_dense_local(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        softcap=softcap,
        score_mod=score_mod,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        batch_start_index=0,
        offset_q=0,
        offset_k=0,
        extra_keep_mask=extra_keep_mask,
        return_lse=return_lse,
    )
    return shim_out, shim_lse, block_size


def compat_replay_varlen_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor | None,
    dlse: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    page_table: torch.Tensor | None,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int | None, int | None],
    learnable_sink: torch.Tensor | None,
    softcap: float | None,
    score_mod: Any,
    mask_mod: Any = None,
    aux_tensors: list[torch.Tensor] | None = None,
    block_sparse_tensors: Any = None,
    block_size: tuple[int, int] | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_req = q.detach().clone().requires_grad_(True)
        k_req = k.detach().clone().requires_grad_(True)
        v_req = v.detach().clone().requires_grad_(True)
        shim_out, shim_lse = attention_forward_varlen_local(
            q_req,
            k_req,
            v_req,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=0.0 if softcap is None else float(softcap),
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=block_sparse_tensors,
            block_size=block_size,
            return_lse=return_lse,
        )

        outputs: list[torch.Tensor] = [shim_out]
        grad_outputs: list[torch.Tensor] = [
            torch.zeros_like(shim_out) if dout is None else _convert_tensor_like(dout, shim_out)
        ]
        if return_lse and shim_lse is not None:
            outputs.append(shim_lse)
            if dlse is None:
                grad_outputs.append(torch.zeros_like(shim_lse))
            else:
                grad_outputs.append(_convert_tensor_like(dlse, shim_lse))

        dq_grad, dk_grad, dv_grad = torch.autograd.grad(
            outputs=tuple(outputs),
            inputs=(q_req, k_req, v_req),
            grad_outputs=tuple(grad_outputs),
            retain_graph=False,
            allow_unused=False,
        )
    return dq_grad.detach(), dk_grad.detach(), dv_grad.detach()


class NativeProbeForwardBridge(JitCompiledFunction):
    def __init__(self, kernel: Any):
        self.kernel = kernel
        self.kernel_name = type(kernel).__name__

    def __repr__(self) -> str:
        dense_status = native_dense_backend_status()
        varlen_status = native_varlen_backend_status()
        dense_backend = "compiled" if dense_status.get("loaded") else "fallback"
        varlen_backend = "compiled" if varlen_status.get("loaded") else "fallback"
        return (
            f"<NativeProbeForwardBridge {self.kernel_name} "
            f"dense_backend={dense_backend} varlen_backend={varlen_backend}>"
        )

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
        window_size = _window_tuple(window_size_left, window_size_right)
        score_mod = getattr(self.kernel, "score_mod", None)
        mask_mod = getattr(self.kernel, "mask_mod", None)
        softcap, score_mod = _extract_softcap_score_mod_components(score_mod)
        shim_out, shim_lse, block_size = _run_forward_with_bridge_runtime(
            kernel=self.kernel,
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap or 0.0,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=lse is not None,
        )
        _copy_tensor_like(out, shim_out)
        _copy_tensor_like(lse, shim_lse)
        _stash_forward_metadata(
            out,
            lse,
            softcap=softcap or 0.0,
            score_mod=score_mod,
            mask_mod=mask_mod,
            learnable_sink=learnable_sink,
            aux_tensors=aux_tensors,
            page_table=page_table,
            block_sparse_tensors=block_sparse_tensors,
            block_size=block_size,
        )


class NativeProbeBackwardPreprocessBridge(JitCompiledFunction):
    def __repr__(self) -> str:
        status = native_bwd_helpers_backend_status()
        backend = "compiled" if status.get("loaded") else "fallback"
        return f"<NativeProbeBackwardPreprocessBridge backend={backend}>"

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
        del dout, cu_seqlens_q, seqused_q
        try:
            flash_attn_backward_preprocess_zero_native(dpsum, lse_log2, dq_accum)
        except Exception:
            if dpsum is not None:
                dpsum.zero_()
            if lse_log2 is not None:
                lse_log2.zero_()
            if dq_accum is not None:
                dq_accum.zero_()
        if dq_accum is not None:
            _DLSE_BY_DQACCUM[dq_accum.data_ptr()] = dlse.detach().clone() if dlse is not None else None
            metadata = _pop_forward_metadata(out, lse)
            if metadata is not None:
                _BWD_METADATA_BY_DQACCUM[dq_accum.data_ptr()] = metadata


class NativeProbeBackwardPostprocessBridge(JitCompiledFunction):
    def __repr__(self) -> str:
        status = native_bwd_helpers_backend_status()
        backend = "compiled" if status.get("loaded") else "fallback"
        return f"<NativeProbeBackwardPostprocessBridge backend={backend}>"

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
        try:
            flash_attn_backward_postprocess_copy_native(result, output)
        except Exception:
            _copy_tensor_like(output, result)


class NativeProbeBackwardBridge(JitCompiledFunction):
    def __init__(self, kernel: Any):
        self.kernel = kernel
        self.kernel_name = type(kernel).__name__

    def __repr__(self) -> str:
        dense_status = native_dense_backend_status()
        varlen_status = native_varlen_backend_status()
        dense_backend = "compiled" if dense_status.get("loaded") else "fallback"
        varlen_backend = "compiled" if varlen_status.get("loaded") else "fallback"
        return (
            f"<NativeProbeBackwardBridge {self.kernel_name} "
            f"dense_backend={dense_backend} varlen_backend={varlen_backend}>"
        )

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
        dlse = _DLSE_BY_DQACCUM.pop(dq_accum.data_ptr(), None)
        compat_metadata = _BWD_METADATA_BY_DQACCUM.pop(dq_accum.data_ptr(), {})
        kernel_score_mod = getattr(self.kernel, "score_mod", None)
        composed_softcap, _ = _extract_softcap_score_mod_components(kernel_score_mod)
        softcap_value = 0.0 if softcap is None else float(softcap)
        if composed_softcap is not None:
            # The backward kernel already carries a score_mod wrapper that folds
            # softcap into the score transform, so replay should not re-apply it.
            softcap_value = 0.0
        elif softcap_value == 0.0:
            softcap_value = float(compat_metadata.get("softcap", 0.0) or 0.0)
        with torch.enable_grad():
            q_req = q.detach().clone().requires_grad_(True)
            k_req = k.detach().clone().requires_grad_(True)
            v_req = v.detach().clone().requires_grad_(True)
            window_size = _window_tuple(window_size_left, window_size_right)
            score_mod = kernel_score_mod or compat_metadata.get("score_mod")
            mask_mod = getattr(self.kernel, "mask_mod", None) or compat_metadata.get("mask_mod")
            learnable_sink = compat_metadata.get("learnable_sink")
            if aux_tensors is None:
                aux_tensors = compat_metadata.get("aux_tensors")
            page_table = compat_metadata.get("page_table")
            compat_block_sparse_tensors = compat_metadata.get("block_sparse_tensors")
            compat_block_size = compat_metadata.get("block_size")
            if compat_block_sparse_tensors is not None:
                block_sparse_tensors = compat_block_sparse_tensors
            elif block_sparse_tensors is not None:
                raise NotImplementedError(
                    "Block-sparse native probe backward requires forward metadata from the bridged forward path"
                )
            if block_sparse_tensors is not None and compat_block_size is None:
                raise NotImplementedError(
                    "Block-sparse native probe backward bridge is missing the original block size metadata"
                )
            shim_out, shim_lse, _ = _run_forward_with_bridge_runtime(
                kernel=self.kernel,
                q=q_req,
                k=k_req,
                v=v_req,
                softmax_scale=softmax_scale,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                window_size=window_size,
                learnable_sink=learnable_sink,
                softcap=softcap_value,
                score_mod=score_mod,
                mask_mod=mask_mod,
                aux_tensors=aux_tensors,
                block_sparse_tensors=block_sparse_tensors,
                block_size_override=compat_block_size,
                return_lse=dlse is not None,
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


class NativeCompiledForwardCombineBridge(JitCompiledFunction):
    def __repr__(self) -> str:
        status = native_combine_backend_status()
        backend = "compiled" if status.get("loaded") else "fallback"
        return f"<NativeCompiledForwardCombineBridge backend={backend}>"

    def __call__(
        self,
        out_partial: torch.Tensor,
        lse_partial: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        seqused: torch.Tensor | None,
        num_splits_dynamic_ptr: torch.Tensor | None,
        varlen_batch_idx: torch.Tensor | None,
        semaphore_to_reset: torch.Tensor | None,
    ) -> None:
        del semaphore_to_reset
        shim_out, shim_lse = flash_attn_combine_native(
            out_partial,
            lse_partial,
            out_dtype=out.dtype,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            varlen_batch_idx=varlen_batch_idx,
            num_splits_dynamic_ptr=num_splits_dynamic_ptr,
            return_lse=lse is not None,
        )
        _copy_tensor_like(out, shim_out)
        _copy_tensor_like(lse, shim_lse)


class NativeProbeBlockSparsityBridge(JitCompiledFunction):
    def __init__(self, kernel: Any):
        self.kernel = kernel

    def __repr__(self) -> str:
        return "<NativeProbeBlockSparsityBridge>"

    def __call__(
        self,
        blocksparse_tensors_torch,
        seqlen_q: int | torch.Tensor,
        seqlen_k: int | torch.Tensor,
        aux_tensors: list[torch.Tensor] | None,
    ) -> None:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = blocksparse_tensors_torch
        device = mask_block_cnt.device
        batch_size, num_heads, _ = mask_block_cnt.shape
        fill_block_sparse_tensors(
            tile_m=int(self.kernel.tile_mn[0]),
            tile_n=int(self.kernel.tile_mn[1]),
            batch_size=int(batch_size),
            num_heads=int(num_heads),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            mask_mod=self.kernel.mask_mod,
            aux_tensors=aux_tensors,
            device=device,
            compute_full_blocks=bool(getattr(self.kernel, "compute_full_blocks", True)),
            use_fast_sampling=bool(getattr(self.kernel, "use_fast_sampling", False)),
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
        )


class NativeProbeUnsupportedBridge(JitCompiledFunction):
    def __init__(self, op: Any):
        self.op = op
        self.op_name = type(op).__name__

    def __repr__(self) -> str:
        return f"<NativeProbeUnsupportedBridge {self.op_name}>"

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(f"Native probe compile bridge does not yet implement {self.op_name}")


def compile_dispatch(op: Any, *args, **kwargs):
    del args, kwargs
    op_name = type(op).__name__
    if op_name.startswith("FlashAttentionForwardSm"):
        return NativeProbeForwardBridge(op)
    if op_name == "FlashAttentionForwardCombine":
        return NativeCompiledForwardCombineBridge()
    if op_name == "BlockSparsityKernel":
        return NativeProbeBlockSparsityBridge(op)
    if op_name == "FlashAttentionBackwardPreprocess":
        return NativeProbeBackwardPreprocessBridge()
    if op_name == "FlashAttentionBackwardPostprocess":
        return NativeProbeBackwardPostprocessBridge()
    if op_name.startswith("FlashAttentionBackwardSm"):
        return NativeProbeBackwardBridge(op)
    return NativeProbeUnsupportedBridge(op)


def compile_repr() -> ProbePlaceholder:
    return ProbePlaceholder("cutlass.cute.compile")
