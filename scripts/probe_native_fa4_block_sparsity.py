"""Run block-sparsity parity probes through the isolated native FA4 bridge path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_windows_shim_module():
    repo_root = _repo_root()
    shim_init = repo_root / "shims" / "flash_attn" / "cute" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "native_probe_windows_flash_attn_cute_shim_block_sparse_probe",
        shim_init,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Windows shim module from {shim_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_diff(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
    if actual is None or expected is None:
        print(f"{name}_present={bool(actual is not None and expected is not None)}")
        return
    diff = (actual.float() - expected.float()).abs()
    print(f"{name}_max_diff={diff.max().item()}")
    print(f"{name}_mean_diff={diff.mean().item()}")
    print(f"{name}_exact={bool(diff.max().item() == 0.0)}")


def _clear_compile_cache(module) -> None:
    compile_cache = getattr(module.compute_block_sparsity, "compile_cache", None)
    if hasattr(compile_cache, "clear"):
        compile_cache.clear()


def _run_exact(module, shim_mod):
    def causal_block_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, head_idx, seqlen_info, aux_tensors
        return kv_idx <= q_idx

    _, native_tensors = module.compute_block_sparsity(
        4,
        4,
        1,
        1,
        8,
        8,
        causal_block_mask,
        None,
        torch.device("cuda"),
        compute_full_blocks=True,
    )
    _, ref_tensors = shim_mod.compute_block_sparsity(
        4,
        4,
        1,
        1,
        8,
        8,
        causal_block_mask,
        None,
        torch.device("cuda"),
        compute_full_blocks=True,
    )
    print("case=block_sparse_exact")
    _tensor_diff("block_sparse_exact_mask_cnt", native_tensors.mask_block_cnt, ref_tensors.mask_block_cnt)
    _tensor_diff("block_sparse_exact_mask_idx", native_tensors.mask_block_idx, ref_tensors.mask_block_idx)
    _tensor_diff("block_sparse_exact_full_cnt", native_tensors.full_block_cnt, ref_tensors.full_block_cnt)
    _tensor_diff("block_sparse_exact_full_idx", native_tensors.full_block_idx, ref_tensors.full_block_idx)


def _run_fast_sampling(module, shim_mod):
    @shim_mod.fast_sampling
    def head_bias_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del batch_idx, seqlen_info
        return kv_idx <= (q_idx + aux_tensors[0][head_idx])

    head_bias = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    _, native_tensors = module.compute_block_sparsity(
        2,
        2,
        1,
        2,
        4,
        4,
        head_bias_mask,
        [head_bias],
        torch.device("cuda"),
        compute_full_blocks=True,
        use_fast_sampling=True,
    )
    _, ref_tensors = shim_mod.compute_block_sparsity(
        2,
        2,
        1,
        2,
        4,
        4,
        head_bias_mask,
        [head_bias],
        torch.device("cuda"),
        compute_full_blocks=True,
        use_fast_sampling=True,
    )
    print("case=block_sparse_fast_sampling")
    _tensor_diff("block_sparse_fast_mask_cnt", native_tensors.mask_block_cnt, ref_tensors.mask_block_cnt)
    _tensor_diff("block_sparse_fast_mask_idx", native_tensors.mask_block_idx, ref_tensors.mask_block_idx)
    _tensor_diff("block_sparse_fast_full_cnt", native_tensors.full_block_cnt, ref_tensors.full_block_cnt)
    _tensor_diff("block_sparse_fast_full_idx", native_tensors.full_block_idx, ref_tensors.full_block_idx)


def main() -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "native_probe_shims"))

    import cutlass
    import flash_attn.cute.compute_block_sparsity as block_sparse_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native block-sparsity probe")

    shim_mod = _load_windows_shim_module()
    print(f"cutlass_probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<unknown>')}")
    print(f"cutlass_probe_reason={getattr(cutlass, 'NATIVE_PROBE_REASON', '<unknown>')}")

    for runner in (
        lambda: _run_exact(block_sparse_module, shim_mod),
        lambda: _run_fast_sampling(block_sparse_module, shim_mod),
    ):
        _clear_compile_cache(block_sparse_module)
        runner()
        compiled_values = list(block_sparse_module.compute_block_sparsity.compile_cache.values())
        compiled_types = (
            sorted({type(value).__name__ for value in compiled_values})
            if compiled_values
            else ["missing"]
        )
        print(f"compiled_types={compiled_types}")
        if compiled_values:
            print(f"compiled_repr_sample={repr(compiled_values[0])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
