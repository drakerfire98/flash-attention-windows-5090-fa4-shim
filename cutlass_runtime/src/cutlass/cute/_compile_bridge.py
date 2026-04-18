"""Runtime-local FA4 compile bridge for the Windows CuTe probe path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from _probe_helpers import ProbePlaceholder
from cutlass.cutlass_dsl import JitCompiledFunction
from ._native_backend import flash_attn_combine_native, native_combine_backend_status
from ._native_bwd_helpers_backend import (
    flash_attn_backward_postprocess_copy_native,
    flash_attn_backward_preprocess_zero_native,
    native_bwd_helpers_backend_status,
)
from ._native_dense_bwd_backend import (
    flash_attn_dense_backward_native,
    native_dense_bwd_backend_status,
)
from ._native_dense_backend import flash_attn_dense_forward_native, native_dense_backend_status
from ._native_varlen_backend import flash_attn_varlen_forward_native, native_varlen_backend_status
from ._native_varlen_bwd_backend import (
    flash_attn_varlen_backward_native,
    native_varlen_bwd_backend_status,
)
from ._runtime_local_core import (
    attention_forward_dense_local,
    attention_forward_varlen_local,
    fill_block_sparse_tensors,
    materialize_dense_keep_mask,
    materialize_dense_score_bias,
    materialize_varlen_keep_mask,
    materialize_varlen_score_bias,
)


_DLSE_BY_DQACCUM: dict[int, torch.Tensor | None] = {}
_POSTPROCESS_RESULTS: dict[int, torch.Tensor] = {}
_FWD_METADATA_BY_TENSOR_PTR: dict[int, dict[str, Any]] = {}
_BWD_METADATA_BY_DQACCUM: dict[int, dict[str, Any]] = {}


@dataclass(frozen=True)
class NativePlanSpec:
    family: str
    op_name: str
    runtime_mode: str
    backend_summary: str
    modifier_family: str
    native_support: str
    compile_surface: str = "cutlass.cute.compile"

    def as_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "op_name": self.op_name,
            "runtime_mode": self.runtime_mode,
            "backend_summary": self.backend_summary,
            "modifier_family": self.modifier_family,
            "native_support": self.native_support,
            "compile_surface": self.compile_surface,
        }


class NativeCompiledPlan(JitCompiledFunction):
    def __init__(self, spec: NativePlanSpec, executor: Callable[..., Any]):
        self.spec = spec
        self._executor = executor

    def __call__(self, *args, **kwargs):
        return self._executor(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<NativeCompiledPlan family={self.spec.family} op={self.spec.op_name} "
            f"modifier={self.spec.modifier_family} support={self.spec.native_support} "
            f"runtime={self.spec.runtime_mode} backend={self.spec.backend_summary}>"
        )

    def export_to_c(self, *args, **kwargs):
        object_file_path = kwargs.get("object_file_path")
        function_name = kwargs.get("function_name", "func")
        if object_file_path is None:
            raise ValueError("NativeCompiledPlan.export_to_c requires object_file_path=...")
        manifest = {
            "format": "native-compiled-plan-v1",
            "function_name": str(function_name),
            "spec": self.spec.as_dict(),
            "executor": _serialize_executor(self._executor, self.spec),
        }
        Path(object_file_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class NativeRuntimeCompiler:
    def __call__(self, op: Any, *args, **kwargs):
        return compile_dispatch(op, *args, **kwargs)

    def __repr__(self) -> str:
        return (
            "<NativeRuntimeCompiler runtime=runtime-local-owned "
            "surface=plan-backed hybrid=compiled-slices+runtime-executors>"
        )


class NativePatchedRuntimeCompiler(NativeRuntimeCompiler):
    def __init__(self, patched_compile: Callable[..., Any]):
        self._patched_compile = patched_compile

    def __call__(self, op: Any, *args, **kwargs):
        return self._patched_compile(op, *args, **kwargs)

    def __repr__(self) -> str:
        return (
            "<NativePatchedRuntimeCompiler runtime=runtime-local-owned "
            "surface=plan-backed cubin-dump-patch=enabled>"
        )


def _dynamic_kernel_type(name: str):
    return type(name, (), {})


def _serialize_executor(executor: Callable[..., Any], spec: NativePlanSpec) -> dict[str, Any]:
    if isinstance(executor, NativeProbeForwardBridge):
        return {
            "kind": "forward_bridge",
            "kernel": _serialize_kernel_payload(executor.kernel, spec),
        }
    if isinstance(executor, NativeProbeBackwardBridge):
        return {
            "kind": "backward_bridge",
            "kernel": _serialize_kernel_payload(executor.kernel, spec),
        }
    if isinstance(executor, NativeProbeBackwardPreprocessBridge):
        return {"kind": "backward_preprocess"}
    if isinstance(executor, NativeProbeBackwardPostprocessBridge):
        return {"kind": "backward_postprocess"}
    if isinstance(executor, NativeCompiledForwardCombineBridge):
        return {"kind": "forward_combine"}
    if isinstance(executor, NativeProbeUnsupportedBridge):
        return {"kind": "unsupported", "op_name": executor.op_name}
    raise NotImplementedError(f"Native plan export does not support executor type {type(executor).__name__}")


def _serialize_kernel_payload(kernel: Any, spec: NativePlanSpec) -> dict[str, Any]:
    modifier_family = spec.modifier_family
    score_mod = getattr(kernel, "score_mod", None)
    softcap, nested_score_mod = _extract_softcap_score_mod_components(score_mod)
    if modifier_family in {"keep-mask", "score-bias", "softcap+score-bias", "score-mod-bwd"}:
        raise NotImplementedError(
            f"Persistent export is not yet supported for modifier family {modifier_family!r}"
        )
    if getattr(kernel, "mask_mod", None) is not None:
        raise NotImplementedError("Persistent export does not support callable mask_mod kernels yet")
    if nested_score_mod is not None:
        raise NotImplementedError("Persistent export does not support nested score_mod kernels yet")
    attrs = {
        "is_causal": bool(getattr(kernel, "is_causal", False)),
        "is_local": bool(getattr(kernel, "is_local", False)),
        "qhead_per_kvhead": int(getattr(kernel, "qhead_per_kvhead", 1) or 1),
    }
    for name in ("tile_m", "tile_n", "q_subtile_factor"):
        value = getattr(kernel, name, None)
        if value is not None:
            attrs[name] = int(value)
    if softcap is not None:
        attrs["_native_export_softcap"] = float(softcap)
    return {
        "op_name": type(kernel).__name__,
        "attrs": attrs,
    }


def _hydrate_kernel_payload(payload: dict[str, Any]) -> Any:
    kernel_type = _dynamic_kernel_type(str(payload["op_name"]))
    kernel = kernel_type()
    for name, value in dict(payload.get("attrs", {})).items():
        setattr(kernel, name, value)
    return kernel


def load_exported_native_plan(manifest: dict[str, Any]) -> tuple[str, NativeCompiledPlan]:
    if manifest.get("format") != "native-compiled-plan-v1":
        raise ValueError(f"Unsupported native plan manifest format: {manifest.get('format')!r}")
    spec = NativePlanSpec(**dict(manifest["spec"]))
    executor_manifest = dict(manifest["executor"])
    kind = executor_manifest["kind"]
    if kind == "forward_bridge":
        executor: Callable[..., Any] = NativeProbeForwardBridge(_hydrate_kernel_payload(executor_manifest["kernel"]))
    elif kind == "backward_bridge":
        executor = NativeProbeBackwardBridge(_hydrate_kernel_payload(executor_manifest["kernel"]))
    elif kind == "backward_preprocess":
        executor = NativeProbeBackwardPreprocessBridge()
    elif kind == "backward_postprocess":
        executor = NativeProbeBackwardPostprocessBridge()
    elif kind == "forward_combine":
        executor = NativeCompiledForwardCombineBridge()
    elif kind == "unsupported":
        op_name = str(executor_manifest.get("op_name", spec.op_name))
        executor = NativeProbeUnsupportedBridge(SimpleNamespace(__class__=_dynamic_kernel_type(op_name)))
    else:
        raise ValueError(f"Unsupported native plan executor kind: {kind!r}")
    return str(manifest.get("function_name", "func")), NativeCompiledPlan(spec, executor)
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
        and (softcap is None or float(softcap) == 0.0)
        and block_sparse_tensors is None
    )


def _can_use_native_varlen_backend(
    *,
    is_varlen: bool,
    score_mod: Any,
    mask_mod: Any,
    block_sparse_tensors: Any,
    softcap: float | None,
) -> bool:
    return (
        is_varlen
        and score_mod is None
        and mask_mod is None
        and block_sparse_tensors is None
        and (softcap is None or float(softcap) == 0.0)
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
    structured_score_bias = None

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
            if float(softcap) == 0.0:
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
                    pass
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

    if score_mod is not None and mask_mod is None and block_sparse_tensors is None:
        if is_varlen:
            structured_score_bias = materialize_varlen_score_bias(
                q,
                k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
            )
        else:
            structured_score_bias = materialize_dense_score_bias(
                q,
                k,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
                batch_start_index=0,
                offset_q=0,
                offset_k=0,
            )

    if structured_score_bias is not None and not return_lse and float(softcap) == 0.0:
        if is_varlen:
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
                        extra_score_bias=structured_score_bias,
                        softcap=softcap,
                        return_lse=return_lse,
                    ),
                    block_size,
                )
            except Exception:
                pass
        else:
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
                        extra_score_bias=structured_score_bias,
                        softcap=softcap,
                        return_lse=return_lse,
                    ),
                    block_size,
                )
            except Exception:
                pass

    if _can_use_native_varlen_backend(
        is_varlen=is_varlen,
        score_mod=score_mod,
        mask_mod=mask_mod,
        block_sparse_tensors=block_sparse_tensors,
        softcap=softcap,
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
        explicit_softcap = getattr(self.kernel, "_native_export_softcap", None)
        softcap, score_mod = _extract_softcap_score_mod_components(score_mod)
        if explicit_softcap is not None and softcap is None:
            softcap = float(explicit_softcap)
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
        dense_bwd_status = native_dense_bwd_backend_status()
        varlen_status = native_varlen_backend_status()
        varlen_bwd_status = native_varlen_bwd_backend_status()
        dense_backend = "compiled" if dense_status.get("loaded") else "fallback"
        dense_bwd_backend = "compiled" if dense_bwd_status.get("loaded") else "fallback"
        varlen_backend = "compiled" if varlen_status.get("loaded") else "fallback"
        varlen_bwd_backend = "compiled" if varlen_bwd_status.get("loaded") else "fallback"
        return (
            f"<NativeProbeBackwardBridge {self.kernel_name} "
            f"dense_backend={dense_backend} dense_bwd_backend={dense_bwd_backend} "
            f"varlen_backend={varlen_backend} varlen_bwd_backend={varlen_bwd_backend}>"
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
        explicit_softcap = getattr(self.kernel, "_native_export_softcap", None)
        composed_softcap, _ = _extract_softcap_score_mod_components(kernel_score_mod)
        softcap_value = 0.0 if softcap is None else float(softcap)
        if composed_softcap is not None:
            # The backward kernel already carries a score_mod wrapper that folds
            # softcap into the score transform, so replay should not re-apply it.
            softcap_value = 0.0
        elif explicit_softcap is not None:
            softcap_value = float(explicit_softcap)
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
            is_varlen = any(t is not None for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table))
            structured_score_bias = None
            if score_mod is not None and mask_mod is None and block_sparse_tensors is None and softcap_value == 0.0:
                if is_varlen:
                    structured_score_bias = materialize_varlen_score_bias(
                        q_req,
                        k_req,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        page_table=page_table,
                        score_mod=score_mod,
                        aux_tensors=aux_tensors,
                    )
                else:
                    structured_score_bias = materialize_dense_score_bias(
                        q_req,
                        k_req,
                        score_mod=score_mod,
                        aux_tensors=aux_tensors,
                        batch_start_index=0,
                        offset_q=0,
                        offset_k=0,
                    )
            if not is_varlen and score_mod is None and softcap_value == 0.0:
                extra_keep_mask = None
                if mask_mod is not None or block_sparse_tensors is not None:
                    extra_keep_mask = materialize_dense_keep_mask(
                        q_req,
                        k_req,
                        mask_mod=mask_mod,
                        aux_tensors=aux_tensors,
                        batch_start_index=0,
                        offset_q=0,
                        offset_k=0,
                        block_sparse_tensors=block_sparse_tensors,
                        block_size=compat_block_size,
                    )
                dlse_native = None
                if dlse is not None:
                    expected_lse = torch.empty(
                        (q.shape[0], q.shape[1], q.shape[2]),
                        device=dlse.device,
                        dtype=dlse.dtype,
                    )
                    dlse_native = _convert_tensor_like(dlse, expected_lse)
                try:
                    dq_grad, dk_grad, dv_grad = flash_attn_dense_backward_native(
                        q_req,
                        k_req,
                        v_req,
                        dout,
                        dlse=dlse_native,
                        softmax_scale=softmax_scale,
                        causal=bool(getattr(self.kernel, "is_causal", False)),
                        window_size=window_size,
                        learnable_sink=learnable_sink,
                        extra_keep_mask=extra_keep_mask,
                        softcap=softcap_value,
                    )
                    _POSTPROCESS_RESULTS[dq_accum.data_ptr()] = dq_grad.detach()
                    qhead_per_kvhead = int(getattr(self.kernel, "qhead_per_kvhead", 1))
                    if qhead_per_kvhead > 1:
                        _POSTPROCESS_RESULTS[dk_or_accum.data_ptr()] = dk_grad.detach()
                        _POSTPROCESS_RESULTS[dv_or_accum.data_ptr()] = dv_grad.detach()
                    else:
                        _copy_tensor_like(dk_or_accum, dk_grad.detach())
                        _copy_tensor_like(dv_or_accum, dv_grad.detach())
                    return
                except Exception:
                    pass
            if not is_varlen and structured_score_bias is not None:
                dlse_native = None
                if dlse is not None:
                    expected_lse = torch.empty(
                        (q.shape[0], q.shape[1], q.shape[2]),
                        device=dlse.device,
                        dtype=dlse.dtype,
                    )
                    dlse_native = _convert_tensor_like(dlse, expected_lse)
                try:
                    dq_grad, dk_grad, dv_grad = flash_attn_dense_backward_native(
                        q_req,
                        k_req,
                        v_req,
                        dout,
                        dlse=dlse_native,
                        softmax_scale=softmax_scale,
                        causal=bool(getattr(self.kernel, "is_causal", False)),
                        window_size=window_size,
                        learnable_sink=learnable_sink,
                        extra_score_bias=structured_score_bias,
                        softcap=softcap_value,
                    )
                    _POSTPROCESS_RESULTS[dq_accum.data_ptr()] = dq_grad.detach()
                    qhead_per_kvhead = int(getattr(self.kernel, "qhead_per_kvhead", 1))
                    if qhead_per_kvhead > 1:
                        _POSTPROCESS_RESULTS[dk_or_accum.data_ptr()] = dk_grad.detach()
                        _POSTPROCESS_RESULTS[dv_or_accum.data_ptr()] = dv_grad.detach()
                    else:
                        _copy_tensor_like(dk_or_accum, dk_grad.detach())
                        _copy_tensor_like(dv_or_accum, dv_grad.detach())
                    return
                except Exception:
                    pass
            if is_varlen and score_mod is None and softcap_value == 0.0:
                extra_keep_mask = None
                if mask_mod is not None or block_sparse_tensors is not None:
                    extra_keep_mask = materialize_varlen_keep_mask(
                        q_req,
                        k_req,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        page_table=page_table,
                        mask_mod=mask_mod,
                        aux_tensors=aux_tensors,
                        block_sparse_tensors=block_sparse_tensors,
                        block_size=compat_block_size,
                    )
                dlse_native = None
                if dlse is not None:
                    if cu_seqlens_q is not None:
                        expected_lse = torch.empty(
                            (q.shape[0], q.shape[1]),
                            device=dlse.device,
                            dtype=dlse.dtype,
                        )
                    else:
                        expected_lse = torch.empty(
                            (q.shape[0], q.shape[1], q.shape[2]),
                            device=dlse.device,
                            dtype=dlse.dtype,
                        )
                    dlse_native = _convert_tensor_like(dlse, expected_lse)
                try:
                    dq_grad, dk_grad, dv_grad = flash_attn_varlen_backward_native(
                        q_req,
                        k_req,
                        v_req,
                        dout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        page_table=page_table,
                        dlse=dlse_native,
                        softmax_scale=softmax_scale,
                        causal=bool(getattr(self.kernel, "is_causal", False)),
                        window_size=window_size,
                        learnable_sink=learnable_sink,
                        extra_keep_mask=extra_keep_mask,
                        softcap=softcap_value,
                    )
                    _POSTPROCESS_RESULTS[dq_accum.data_ptr()] = dq_grad.detach()
                    qhead_per_kvhead = int(getattr(self.kernel, "qhead_per_kvhead", 1))
                    if qhead_per_kvhead > 1:
                        _POSTPROCESS_RESULTS[dk_or_accum.data_ptr()] = dk_grad.detach()
                        _POSTPROCESS_RESULTS[dv_or_accum.data_ptr()] = dv_grad.detach()
                    else:
                        _copy_tensor_like(dk_or_accum, dk_grad.detach())
                        _copy_tensor_like(dv_or_accum, dv_grad.detach())
                    return
                except Exception:
                    pass
            if is_varlen and structured_score_bias is not None:
                dlse_native = None
                if dlse is not None:
                    if cu_seqlens_q is not None:
                        expected_lse = torch.empty(
                            (q.shape[0], q.shape[1]),
                            device=dlse.device,
                            dtype=dlse.dtype,
                        )
                    else:
                        expected_lse = torch.empty(
                            (q.shape[0], q.shape[1], q.shape[2]),
                            device=dlse.device,
                            dtype=dlse.dtype,
                        )
                    dlse_native = _convert_tensor_like(dlse, expected_lse)
                try:
                    dq_grad, dk_grad, dv_grad = flash_attn_varlen_backward_native(
                        q_req,
                        k_req,
                        v_req,
                        dout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        seqused_q=seqused_q,
                        seqused_k=seqused_k,
                        page_table=page_table,
                        dlse=dlse_native,
                        softmax_scale=softmax_scale,
                        causal=bool(getattr(self.kernel, "is_causal", False)),
                        window_size=window_size,
                        learnable_sink=learnable_sink,
                        extra_score_bias=structured_score_bias,
                        softcap=softcap_value,
                    )
                    _POSTPROCESS_RESULTS[dq_accum.data_ptr()] = dq_grad.detach()
                    qhead_per_kvhead = int(getattr(self.kernel, "qhead_per_kvhead", 1))
                    if qhead_per_kvhead > 1:
                        _POSTPROCESS_RESULTS[dk_or_accum.data_ptr()] = dk_grad.detach()
                        _POSTPROCESS_RESULTS[dv_or_accum.data_ptr()] = dv_grad.detach()
                    else:
                        _copy_tensor_like(dk_or_accum, dk_grad.detach())
                        _copy_tensor_like(dv_or_accum, dv_grad.detach())
                    return
                except Exception:
                    pass
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


def _plan_backend_summary(op_name: str) -> str:
    if op_name.startswith("FlashAttentionForwardSm"):
        dense_status = native_dense_backend_status()
        varlen_status = native_varlen_backend_status()
        dense_backend = "compiled" if dense_status.get("built") else "fallback"
        varlen_backend = "compiled" if varlen_status.get("built") else "fallback"
        return f"dense={dense_backend},varlen={varlen_backend},runtime=owned"
    if op_name.startswith("FlashAttentionBackwardSm"):
        dense_status = native_dense_bwd_backend_status()
        varlen_status = native_varlen_bwd_backend_status()
        dense_backend = "compiled" if dense_status.get("built") else "fallback"
        varlen_backend = "compiled" if varlen_status.get("built") else "fallback"
        return f"dense_bwd={dense_backend},varlen_bwd={varlen_backend},runtime=owned"
    if op_name == "FlashAttentionForwardCombine":
        combine_status = native_combine_backend_status()
        combine_backend = "compiled" if combine_status.get("built") else "fallback"
        return f"combine={combine_backend}"
    if op_name in {"FlashAttentionBackwardPreprocess", "FlashAttentionBackwardPostprocess"}:
        helpers_status = native_bwd_helpers_backend_status()
        helpers_backend = "compiled" if helpers_status.get("built") else "fallback"
        return f"helpers={helpers_backend}"
    if op_name == "BlockSparsityKernel":
        return "runtime=owned"
    return "unsupported"


def _plan_modifier_family(op: Any) -> str:
    op_name = type(op).__name__
    if op_name == "FlashAttentionForwardCombine":
        return "combine"
    if op_name in {"FlashAttentionBackwardPreprocess", "FlashAttentionBackwardPostprocess"}:
        return "helper"
    if op_name == "BlockSparsityKernel":
        return "block-sparsity"

    mask_mod = getattr(op, "mask_mod", None)
    if mask_mod is not None:
        return "keep-mask"

    score_mod = getattr(op, "score_mod", None)
    if score_mod is not None:
        softcap, nested_score_mod = _extract_softcap_score_mod_components(score_mod)
        if softcap is not None and nested_score_mod is not None:
            return "softcap+score-bias"
        if softcap is not None:
            return "softcap"
        return "score-bias"

    if getattr(op, "score_mod_bwd", None) is not None:
        return "score-mod-bwd"
    if bool(getattr(op, "is_local", False)):
        return "local-window"
    if int(getattr(op, "qhead_per_kvhead", 1) or 1) > 1:
        return "gqa-reduction"
    return "plain"


def _plan_native_support(op: Any) -> str:
    op_name = type(op).__name__
    modifier_family = _plan_modifier_family(op)
    if op_name == "FlashAttentionForwardCombine":
        return "compiled"
    if op_name in {"FlashAttentionBackwardPreprocess", "FlashAttentionBackwardPostprocess"}:
        return "compiled"
    if op_name == "BlockSparsityKernel":
        return "runtime-owned"
    if op_name.startswith("FlashAttentionForwardSm"):
        if modifier_family in {"plain", "local-window", "softcap", "gqa-reduction"}:
            return "compiled-candidate"
        if modifier_family == "keep-mask":
            return "compiled-keep-mask-candidate"
        if modifier_family == "score-bias":
            return "compiled-score-bias-candidate"
        if modifier_family == "softcap+score-bias":
            return "out-only-compiled-score-bias-candidate"
        return "runtime-replay"
    if op_name.startswith("FlashAttentionBackwardSm"):
        if modifier_family in {"plain", "local-window", "gqa-reduction"}:
            return "compiled-candidate"
        if modifier_family == "keep-mask":
            return "compiled-keep-mask-candidate"
        if modifier_family == "score-bias":
            return "compiled-score-bias-candidate"
        if modifier_family == "softcap+score-bias":
            return "runtime-replay"
        return "runtime-replay"
    return "unsupported"


def _build_native_plan(op: Any, *args, **kwargs) -> NativeCompiledPlan:
    del args, kwargs
    op_name = type(op).__name__
    family = "unsupported"
    executor: Callable[..., Any]
    if op_name.startswith("FlashAttentionForwardSm"):
        family = "flash_attn_forward"
        executor = NativeProbeForwardBridge(op)
    elif op_name == "FlashAttentionForwardCombine":
        family = "flash_attn_forward_combine"
        executor = NativeCompiledForwardCombineBridge()
    elif op_name == "BlockSparsityKernel":
        family = "block_sparsity"
        executor = NativeProbeBlockSparsityBridge(op)
    elif op_name == "FlashAttentionBackwardPreprocess":
        family = "flash_attn_backward_preprocess"
        executor = NativeProbeBackwardPreprocessBridge()
    elif op_name == "FlashAttentionBackwardPostprocess":
        family = "flash_attn_backward_postprocess"
        executor = NativeProbeBackwardPostprocessBridge()
    elif op_name.startswith("FlashAttentionBackwardSm"):
        family = "flash_attn_backward"
        executor = NativeProbeBackwardBridge(op)
    else:
        executor = NativeProbeUnsupportedBridge(op)
    spec = NativePlanSpec(
        family=family,
        op_name=op_name,
        runtime_mode="runtime-local-owned",
        backend_summary=_plan_backend_summary(op_name),
        modifier_family=_plan_modifier_family(op),
        native_support=_plan_native_support(op),
    )
    return NativeCompiledPlan(spec, executor)


def compile_dispatch(op: Any, *args, **kwargs):
    return _build_native_plan(op, *args, **kwargs)


def compile_repr() -> NativeRuntimeCompiler:
    return NativeRuntimeCompiler()
