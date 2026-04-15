# FA4 Windows Status

This note captures the current state of the FlashAttention 4 attempt on the same RTX 5090 Windows machine.

## Goal

Verify this import and runtime path:

```python
import torch
from flash_attn.cute import flash_attn_func
```

## Dedicated Test Env

- env: `.venv_fa4`
- python: `3.13.9`
- torch: `2.12.0.dev20260414+cu130`
- gpu: `NVIDIA GeForce RTX 5090`

## What Worked

These installed in the isolated FA4 env:

- `numpy==2.4.4`
- `einops==0.8.2`
- `apache-tvm-ffi==0.1.10`
- `torch-c-dlpack-ext==0.1.5`
- `quack-kernels==0.3.10 --no-deps`
- `cuda-python==13.2.0`
- `cuda-bindings==13.2.0`
- `nvidia-cutlass-dsl==4.4.2 --no-deps`
- editable install of `flash-attn-4` from `third_party/flash-attention-for-windows/flash_attn/cute`

## What Fails In A Clean Import

The first real FA4 import still failed:

```python
from flash_attn.cute import flash_attn_func
```

Observed error:

```text
ImportError: cannot import name '__version__' from 'cuda' (unknown location)
```

That happens before the code even gets to `cutlass.cute`.

## Native Blockers We Confirmed

There are now two confirmed native blockers on this Windows stack.

1. CUTLASS expects the older top-level CUDA Python API shape.

The linked CUTLASS Python tree imports:

```python
from cuda import __version__
from cuda import cuda, cudart, nvrtc
```

But the installed Windows CUDA Python packages expose a namespace layout under:

- `cuda.bindings.driver`
- `cuda.bindings.runtime`
- `cuda.bindings.nvrtc`

So a clean `import cutlass` fails immediately on `from cuda import __version__`.

2. The linked CUTLASS Python tree still does not contain `cutlass.cute`.

The editable CUTLASS tree that becomes importable after a temporary compatibility patch lives at:

- `third_party/flash-attention-for-windows/csrc/cutlass/python/cutlass`

That tree exists, but it contains entries like:

- `backend/`
- `emit/`
- `epilogue/`
- `op/`
- `shape.py`
- `swizzle.py`

and it does **not** contain:

- `cutlass/cute/`

So even after forcing a temporary in-memory CUDA compatibility shim that allows `import cutlass`, the next native failure is still:

```text
ModuleNotFoundError: No module named 'cutlass.cute'
```

There is also a separate `pycute` tree in the upstream CUTLASS source, but that is not the import path `flash_attn.cute` currently targets.

## Shim Progress

A repo-local compatibility layer was added under `shims/` to push the Windows FA4 path farther without modifying site-packages:

- `shims/cutlass/__init__.py`
- `shims/fcntl.py`
- `shims/flash_attn/__init__.py`
- `shims/flash_attn/cute/__init__.py`
- `scripts/test_fa4_windows_shim.py`
- `scripts/validate_fa4_windows_shim.py`

This shim now does all of the following inside `.venv_fa4`:

- bypasses the original `ModuleNotFoundError: No module named 'cutlass'`
- bypasses the Unix-only `fcntl` import used by `quack.cache_utils`
- shadows the `flash_attn.cute` import path with a stable Windows fallback
- makes this import succeed:

```python
from flash_attn.cute import flash_attn_func
```

## Shim Runtime Result

Using:

```powershell
.\.venv_fa4\Scripts\python.exe .\scripts\test_fa4_windows_shim.py --run-forward
```

Observed behavior:

- import succeeded
- the shimmed module path is `shims/flash_attn/cute/__init__.py`
- a tiny CUDA BF16 forward call returned a CUDA tensor with shape `(1, 64, 4, 64)`
- the return value shape is `(tensor, None)` like the upstream API
- output stayed finite

Dense-path verification is now stable:

- with the seeded probe in `scripts/test_fa4_windows_shim.py --run-forward`, max diff vs `torch.nn.functional.scaled_dot_product_attention` is `0.0`
- mean diff vs SDPA is `0.0`
- with `scripts/test_fa4_windows_shim.py --run-backward`, max grad diffs vs SDPA are all `0.0`

Additional validation coverage now passes in `scripts/validate_fa4_windows_shim.py`:

- local window attention
- dense `softcap`, including backward parity
- dense `mask_mod`, including fully masked-row stability
- dense `mask_mod` backward parity against the manual eager reference
- dense `mask_mod` also validated for `info`-named seqlen modifier signatures
- MQA / GQA style KV head expansion
- `learnable_sink`
- varlen output / LSE reconstruction via chunked dense equivalence
- varlen `softcap`, including backward parity
- varlen `seqused_k` truncation, including backward parity
- varlen `seqused_q` padded-Q support, with zero-filled padded outputs and `-inf` padded LSE rows
- mixed varlen layout coverage for all four upstream-style combinations:
  - packed Q + packed KV
  - padded Q + padded KV
  - packed Q + padded KV
  - padded Q + packed KV
- varlen `score_mod`, including global offset-aware `seqlen_info.offset_k`
- mixed padded varlen `score_mod` coverage using logical global offsets on both Q and KV
- varlen `score_mod` backward parity against the manual eager reference
- varlen aux-only `score_mod(..., aux_tensors)` callable form
- varlen `score_mod` also validated for `info`-named seqlen modifier signatures

Varlen note:

- a naive varlen comparison against raw SDPA can look wrong when `seqlen_q != seqlen_k` because FlashAttention uses bottom-right causal alignment semantics
- the shimmed varlen path was cross-checked against the installed FA2 implementation on the same machine
- observed cross-check:
  - output max diff vs FA2: `0.0078125`
  - LSE diff after transpose alignment: about `2.38e-07`

So the shim is now good for a stable Windows fallback on the public FA4 entrypoints we implemented, but it is still **not** proof of a native CuTe/CUTLASS FA4 kernel path on Windows.

## Practical Meaning

At the moment this repo contains:

- a verified FA2 Windows path
- a partially assembled FA4 test env
- a precise native FA4 blocker chain:
  - CUDA Python API-shape mismatch
  - then missing `cutlass.cute` even after compatibility patching
- a repo-local FA4 shim path that provides stable dense and varlen public-entrypoint fallbacks on Windows, including `learnable_sink`, dense `mask_mod`, varlen `score_mod`, `seqused_q`, and the mixed packed/padded varlen layouts

It does **not** yet contain a verified native Windows FA4 runtime path.

## Best Next Step

The next promising route is one of:

1. Build or locate a Windows CUDA compatibility layer that exposes the old `from cuda import __version__, cuda, cudart, nvrtc` surface cleanly.
2. Build or locate a CUTLASS Python package that actually provides `cutlass.cute` on Windows, not just the adjacent CUTLASS / pycute trees.
3. Extend the shim only when more FA4 surface area is actually needed, keeping unsupported features explicit instead of silently approximated.
4. Re-test native FA4 when upstream or community Windows packaging for the CUTLASS DSL layer matures.
