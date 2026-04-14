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
- `nvidia-cutlass-dsl==4.4.2 --no-deps`
- editable install of `flash-attn-4` from `third_party/flash-attention-for-windows/flash_attn/cute`

## What Failed First

The first real FA4 import still failed:

```python
from flash_attn.cute import flash_attn_func
```

Observed error:

```text
ModuleNotFoundError: No module named 'cutlass'
```

## Why It Failed

`nvidia-cutlass-dsl==4.4.2` did install, but only as metadata.

`pip show -f nvidia-cutlass-dsl` reported only:

- `nvidia_cutlass_dsl-4.4.2.dist-info/...`

It did **not** install an actual `cutlass` package tree, which is what `flash_attn.cute` imports immediately.

The package metadata also still declares:

- `Requires: nvidia-cutlass-dsl-libs-base`

That dependency is the unresolved native Windows-side blocker in this environment.

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
- dense `mask_mod`, including fully masked-row stability
- MQA / GQA style KV head expansion
- `learnable_sink`
- varlen output / LSE reconstruction via chunked dense equivalence
- varlen `score_mod`, including global offset-aware `seqlen_info.offset_k`

Varlen note:

- a naive varlen comparison against raw SDPA can look wrong when `seqlen_q != seqlen_k` because FlashAttention uses bottom-right causal alignment semantics
- the shimmed varlen path was cross-checked against the installed FA2 implementation on the same machine
- observed cross-check:
  - output max diff vs FA2: `0.0078125`
  - LSE diff after transpose alignment: about `2.38e-07`

So the shim is now good for a stable Windows fallback on the public FA4 entrypoints we implemented, but it is still **not** proof of a native CuTe/CUTLASS FA4 kernel path on Windows.

## Upstream Support Signal

NVIDIA's current CUTLASS DSL quick-start docs for the latest 4.4 line state that the supported target is Linux. That matches the behavior seen here: the FA4 Python package can be staged, but the actual CuTe DSL runtime layer needed for `cutlass.cute` is not landing as a usable Windows install in this environment.

## Practical Meaning

At the moment this repo contains:

- a verified FA2 Windows path
- a partially assembled FA4 test env
- a precise native FA4 import blocker
- a repo-local FA4 shim path that provides stable dense and varlen public-entrypoint fallbacks on Windows, including `learnable_sink`, dense `mask_mod`, and varlen `score_mod`

It does **not** yet contain a verified native Windows FA4 runtime path.

## Best Next Step

The next promising route is one of:

1. Find a real Windows build of `nvidia-cutlass-dsl-libs-base` that exposes the `cutlass` Python package.
2. Build the missing CUTLASS DSL layer from source in a way that produces `cutlass.cute` for this Windows stack.
3. Extend the shim only when more FA4 surface area is actually needed, keeping unsupported features explicit instead of silently approximated.
4. Re-test native FA4 when upstream or community Windows packaging for the CUTLASS DSL layer matures.
