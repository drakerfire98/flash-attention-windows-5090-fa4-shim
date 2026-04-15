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

Important packaging note:

- `nvidia-cutlass-dsl` is present as the top-level Python distribution metadata in this env
- but the separate runtime packages it declares, such as `nvidia-cutlass-dsl-libs-base==4.4.2`, are not importable on this Windows stack
- querying PyPI from the isolated env for `nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13` returned `No matching distribution found`
- a repo-local editable compat package can now be installed from `runtime_compat/` to restore the missing raw import surface without solving those missing runtime wheels
- a second repo-local editable package can now be installed from `cutlass_runtime/` to expose the repo's native probe `cutlass` package as a normal top-level module in `.venv_fa4`

## What Fails In A Clean Import

The first real FA4 import still failed:

```python
from flash_attn.cute import flash_attn_func
```

Original observed error:

```text
ImportError: cannot import name '__version__' from 'cuda' (unknown location)
```

That happens before the code even gets to `cutlass.cute`.

After installing the repo-local compat package with:

```powershell
.\.venv_fa4\Scripts\python.exe -m pip install -e .\runtime_compat
```

the raw runtime probe now improves to:

- `raw_cutlass_import=ok`
- `raw_cuda_import=ok`
- `raw_nvidia_cutlass_dsl_import=ok`

So the first import-shape mismatch is now stabilized on Windows even though the
deeper native runtime is still incomplete.

After then installing the repo-local cutlass wrapper with:

```powershell
.\.venv_fa4\Scripts\python.exe -m pip install -e .\cutlass_runtime
```

the clean top-level import also advances:

- `raw_cutlass_spec=C:\...\cutlass_runtime\src\cutlass\__init__.py`
- `raw_cutlass_import=ok`
- `import flash_attn.cute` succeeds without manual `sys.path.insert(...)` of `native_probe_shims`

## Native Blockers We Confirmed

There is now one main deeper native blocker on this Windows stack after the
raw import-shape fix and the installable top-level `cutlass` wrapper.

1. The runtime still depends on repo-local probe bridge layers instead of a true compiled Windows CuTe/CUTLASS DSL runtime.

The runtime probe previously reported:

```text
cutlass_probe_mode=legacy-editable-cutlass
```

The probe path now imports `cutlass` through the installable runtime package first and currently reports:

```text
cutlass_probe_mode=runtime-local-core
cute_file=...\cutlass_runtime\src\cutlass\cute\__init__.py
```

That means the environment is now entering through:

- `cutlass_runtime/src/cutlass/__init__.py`
- `cutlass_runtime/src/cutlass/cute/__init__.py`

but deeper execution still ultimately delegates into repo-local bridge pieces such as:

- other repo-local probe subpackages behind the runtime package
- selective wrappers in `native_probe_shims/` that now point back to the runtime-owned implementations for compatibility

rather than from a separately packaged modern Windows CUTLASS runtime. The
new `cutlass_runtime/` package now owns both the top-level `cutlass` root and
the top-level `cutlass.cute` package surface, and it now also owns the heavy
`cutlass.cute._compile_bridge` logic plus the `cutlass.base_dsl.runtime.cuda`
loader path. It still routes recognized kernels through repo-local bridge
layers rather than a true native Windows CuTe/CUTLASS DSL runtime.

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

A second repo-local package now exists under `cutlass_runtime/` to expose the
same native probe `cutlass` surface as a normal top-level package:

- `cutlass_runtime/src/cutlass/__init__.py`
- `cutlass_runtime/src/_probe_helpers.py`
- `cutlass_runtime/src/fcntl.py`

That package keeps the actual implementation in `native_probe_shims/` but makes
these clean imports work in `.venv_fa4` without manual `sys.path` edits:

```python
import cutlass.cute as cute
import flash_attn.cute as fa4
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

- dynamic split-aware forward-combine masking for both batched and varlen layouts
- block-sparse forward parity against the dense mask-mod reference path
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

## Native Probe Progress

A second, isolated probe path now exists under:

- `native_probe_shims/`
- `scripts/probe_cutlass_runtime.py`
- `scripts/probe_native_fa4_import.py`
- `scripts/probe_native_fa4_forward.py`
- `scripts/probe_native_fa4_backward.py`
- `scripts/probe_native_fa4_combine.py`
- `scripts/probe_native_fa4_block_sparsity.py`
- `scripts/patch_flash_attn_sm120_backward.py`

This path is separate from the stable fallback shim under `shims/`. Its only job is to push the
real native `flash_attn.cute` import/runtime chain forward far enough to reveal the next honest
blocker without pretending the backend is fully implemented.

Current probe result:

- `scripts/probe_cutlass_runtime.py` now reports which CUTLASS package the native probe actually selected
- `scripts/probe_cutlass_cubin_loader.py` now also compiles a tiny NVRTC kernel and verifies that the Windows runtime shim can load, resolve, and unload it
- the repo-local editable package under `runtime_compat/` now fixes the raw Windows import mismatch directly:
  - `cuda` becomes importable as a regular package with `__version__`, `cuda`, `cudart`, and `nvrtc`
  - `nvidia_cutlass_dsl` becomes importable as a real top-level module name
- the repo-local editable package under `cutlass_runtime/` now makes raw `cutlass` resolve to:
  - `cutlass_runtime/src/cutlass/__init__.py`
  - while still delegating to the legacy editable CUTLASS tree for the pieces that only exist there
- the probe now auto-prefers a real modern CUTLASS package if one ever becomes importable on this machine
- on the current Windows env, no such modern package is importable, so the probe still falls back to the legacy editable CUTLASS tree plus compatibility shims
- `scripts/probe_native_fa4_import.py` now imports the real upstream `flash_attn.cute` package successfully
- this requires compatibility scaffolding for:
  - the old top-level CUDA Python API shape
  - the missing `cutlass.cute` package tree
  - newer CUTLASS DSL-side package/modules expected by FA4 and `quack`
  - Unix-only `fcntl`
- `scripts/probe_native_fa4_forward.py` now reaches a tiny CUDA forward call through that native path
- recognized FA4 forward-kernel `cute.compile(...)` calls now return a real `NativeProbeForwardBridge` object instead of a dead placeholder
- that bridge routes execution onto the validated Windows shim path for the forward kernel family we currently recognize
- recognized FA4 forward-combine `cute.compile(...)` calls now also return a real `NativeProbeForwardCombineBridge`
- that forward-combine bridge now also handles `num_splits_dynamic_ptr`, with exact parity in both the batched and varlen dynamic-split probe cases
- recognized FA4 `compute_block_sparsity(...)` `cute.compile(...)` calls now also return a real `NativeProbeBlockSparsityBridge`
- the forward bridge now also supports end-to-end block-sparse execution onto the stable Windows shim path
- recognized FA4 backward preprocess, main backward, and backward postprocess `cute.compile(...)` calls now also return bridge objects instead of dead placeholders
- `scripts/probe_native_fa4_backward.py` still reaches dense and varlen backward parity against the stable Windows shim with `0.0` seeded output and grad diffs after the compat package is installed
- the backward bridge now also preserves forward-only feature metadata across the preprocess step so unsupported SM120 backward surfaces can fall back compatibly onto the stable Windows shim
- `native_probe_shims/cutlass/base_dsl/runtime/cuda.py` now uses the CUDA runtime-library path (`cudaLibraryLoadData`, `cudaLibraryGetKernel`, `cudaLibraryUnload`) instead of the failing driver-module path (`cuModuleLoadData`)
- `scripts/patch_flash_attn_sm120_backward.py` now reapplies the local SM120 `dQ_single_wg = False` fix and removes the public SM120 guard cluster idempotently instead of relying on memory

This is more real than the earlier placeholder probe, but it is still not native CuTe codegen yet.

Observed probe output with `return_lse=True`:

- output device: `cuda:0`
- output dtype: `torch.bfloat16`
- output stayed finite
- output max diff vs SDPA: `0.0078125`
- output mean diff vs SDPA: about `3.0e-4`
- output sum: about `134.25`
- SDPA reference sum: about `134.21`
- LSE type: `Tensor`
- compiled kernel cache entry type: `NativeProbeForwardBridge`
- compiled kernel cache entry repr: `<NativeProbeForwardBridge FlashAttentionForwardSm120>`
- CUTLASS probe mode: `runtime-local-core`
- CUTLASS probe reason: the repo-local `cutlass_runtime/` package now owns the top-level `cutlass` root directly; deeper probe subpackages still come from the repo-local Windows probe modules rather than a standalone compiled CUTLASS DSL runtime

Observed backward probe output:

- dense output max diff vs stable shim: `0.0`
- dense grad max diff vs stable shim: `0.0`
- varlen output max diff vs stable shim: `0.0`
- varlen grad max diff vs stable shim: `0.0`

Observed widened modifier probe output:

- dense `softcap` forward:
  - output max diff vs stable shim: `0.0`
  - LSE max diff vs stable shim: `0.0`
- dense `softcap` backward:
  - output max diff vs stable shim: `0.0`
  - grad max diff vs stable shim: `0.0`
- dense `learnable_sink` backward:
  - output max diff vs stable shim: `0.0`
  - grad max diff vs stable shim: `0.0`
- dense `mask_mod` forward:
  - output max diff vs stable shim: `0.0`
  - LSE max diff vs stable shim: `0.0`
- dense `mask_mod` backward:
  - output max diff vs stable shim: `0.0`
  - grad max diff vs stable shim: `0.0`
- dense block-sparse backward:
  - output max diff vs stable shim: `0.0`
  - grad max diff vs stable shim: `0.0`
- varlen `softcap` forward:
  - output max diff vs stable shim: `0.0`
  - LSE max diff vs stable shim: `0.0`
- varlen `seqused_q` / `seqused_k` backward:
  - output max diff vs stable shim: `0.0`
  - grad max diff vs stable shim: `0.0`
- varlen `seqused_q` / `seqused_k` plus `score_mod` backward:
  - output max diff vs stable shim: `0.0`
  - grad max diff vs stable shim: `0.0`

Observed forward-combine probe output:

- batched combine output max diff vs stable shim: `0.0`
- batched combine LSE max diff vs stable shim: `0.0`
- varlen combine output max diff vs stable shim: `0.0`
- varlen combine LSE max diff vs stable shim: `0.0`
- dynamic batched combine output max diff vs stable shim: `0.0`
- dynamic batched combine LSE max diff vs stable shim: `0.0`
- dynamic varlen combine output max diff vs stable shim: `0.0`
- dynamic varlen combine LSE max diff vs stable shim: `0.0`
- `_flash_attn_fwd_combine.compile_cache` now holds `NativeProbeForwardCombineBridge`

Observed block-sparsity probe output:

- exact block-sparsity mask/full count and index tensors match the stable Windows shim exactly
- fast-sampling block-sparsity mask/full count and index tensors also match the stable Windows shim exactly
- `compute_block_sparsity.compile_cache` now holds `NativeProbeBlockSparsityBridge`

Observed public SM120 block-sparse probe output:

- dense block-sparse forward output max diff vs stable shim: `0.0`
- dense block-sparse forward LSE max diff vs stable shim: `0.0`
- dense block-sparse backward output max diff vs stable shim: `0.0`
- dense block-sparse backward grad max diff vs stable shim: `0.0`
- the upstream public `flash_attn_func(...)` wrapper now reaches `NativeProbeForwardBridge` / `NativeProbeBackwardBridge` cleanly on SM120 for this tested block-sparse path

Observed cubin loader probe output:

- `cutlass_spec` resolves through `cutlass_runtime/src/cutlass/__init__.py`
- `cutlass.cute` resolves through `cutlass_runtime/src/cutlass/cute/__init__.py`
- `cutlass.cute._compile_bridge` resolves through `cutlass_runtime/src/cutlass/cute/_compile_bridge.py`
- `runtime_cuda_module` resolves through `cutlass_runtime/src/cutlass/base_dsl/runtime/cuda.py`
- `cudaLibraryLoadData` succeeds on an NVRTC-built cubin
- `cudaLibraryGetKernel` resolves the probe kernel successfully
- `cudaLibraryUnload` succeeds cleanly
- importing `cutlass.cute` through the real probe path no longer loads the external `pycute` package

So the modifier surface is now much cleaner:

- dense `softcap`, `learnable_sink`, and `mask_mod` are stable on the native probe bridge path in both forward and backward parity probes
- varlen `softcap` is stable on the native probe bridge path
- varlen `seqused_q` / `seqused_k` and the mixed `seqused + score_mod` backward path are now also stable on the native probe bridge path
- varlen paged-KV is now stable on the native probe bridge path in both forward and backward parity probes
- varlen `softcap + score_mod` is now stable on the native probe bridge path in both forward and backward parity probes
- the internal varlen block-sparse path now reaches the native probe bridge exactly in forward parity probes, and the repo-local backward replay helper matches the stable shim exactly for the same tensors
- the upstream forward-combine path is now also stable on the native probe bridge path for the tested batched and varlen cases
- the upstream block-sparsity precompute path is now also stable on the native probe bridge path for the tested exact and fast-sampling cases

That means the main forward path is no longer blocked by a dead placeholder for these recognized
kernels, and the low-level Windows cubin hook is no longer dead either. The next real missing piece
is still a usable CuTe DSL compiler/runtime implementation behind:

- `cutlass.cute.compile`
- related runtime / cubin load hooks

Until that exists for this Windows stack, the native probe can import and traverse the FA4 code
path and selectively bridge known forward and backward kernels, but it still will not be a true
native FA4 kernel implementation.

Checkpoint note:

- the current backward checkpoint also relies on one local upstream source fix in `third_party/flash-attention-for-windows/flash_attn/cute/interface.py`
- that fix initializes `dQ_single_wg = False` before the architecture branch so the SM120 backward path does not raise `UnboundLocalError`
- the latest checkpoint also relies on local upstream compat patches in that same file so varlen paged-KV backward can replay through the repo-local shim helper instead of tripping `_flash_attn_bwd`'s dense shape assumptions

## Practical Meaning

At the moment this repo contains:

- a verified FA2 Windows path
- a partially assembled FA4 test env
- a precise native FA4 blocker chain:
  - raw CUDA / `nvidia_cutlass_dsl` import mismatch, now fixed by `runtime_compat/`
  - missing top-level `cutlass.cute`, now fixed by `cutlass_runtime/`
  - direct cubin loading, now fixed by the runtime-library shim path
- upstream forward-combine and block-sparsity compile families, now covered by probe bridges
- direct bridge validation now also covers block-sparse forward execution and dynamic split-aware combine execution
- shim validator coverage now also includes varlen paged-KV and combined varlen `softcap + score_mod`
- the remaining honest blocker: no true end-to-end Windows CuTe DSL compiler/runtime behind that import surface
- the top-level `cutlass` root is now repo-local instead of the legacy editable CUTLASS root
- the top-level `cutlass.cute` package is now also repo-local instead of resolving from `native_probe_shims/`
- the heavy `cutlass.cute._compile_bridge` logic is now repo-local instead of loaded from `native_probe_shims/`
- the `cutlass.base_dsl.runtime.cuda` loader path is now repo-local instead of resolving from `native_probe_shims/`
- a repo-local FA4 shim path that provides stable dense and varlen public-entrypoint fallbacks on Windows, including `learnable_sink`, dense `mask_mod`, varlen `score_mod`, `seqused_q`, and the mixed packed/padded varlen layouts

It does **not** yet contain a verified native Windows FA4 runtime path.

## Best Next Step

The next promising route is one of:

1. Internalize the remaining deeper probe subpackages into `cutlass_runtime/` so fewer imports fall through to `native_probe_shims/`.
2. Build or locate the real CuTe DSL compiler/runtime layer so `cutlass.cute.compile` stops resolving recognized kernels to selective bridge objects.
3. If Windows wheels do not materialize, build the missing CUTLASS DSL runtime from source or vendor the required pieces into a separate Windows-focused bridge layer.
4. Keep both `runtime_compat/` and `cutlass_runtime/` small and installable so other Windows users can quickly reproduce the same honest blocker chain.
5. Extend the shim only when more FA4 surface area is actually needed, keeping unsupported features explicit instead of silently approximated.
6. Re-test native FA4 when upstream or community Windows packaging for the CUTLASS DSL layer matures.
