# flash-attention-windows-5090

Working Windows build notes, patches, and smoke tests for building FlashAttention on an NVIDIA RTX 5090 with PyTorch nightly and CUDA 13.0.

## Status

- FlashAttention 2 source build verified on the recorded Windows x64 host in `docs/KNOWN_GOOD_ENV.md`
- Verified on Python 3.13.9
- Verified on `torch 2.12.0.dev20260306+cu130`
- Verified on CUDA toolkit 13.0
- Verified on `NVIDIA GeForce RTX 5090`
- Verified built package result: `flash_attn 2.8.4`
- Exact environment fingerprint recorded in `docs/KNOWN_GOOD_ENV.md`

This repo preserves the exact working path that compiled and ran a CUDA smoke test on the target machine. The main blocker we had to fix was PyTorch's Windows ninja generation quoting `nvcc` with POSIX-style single quotes, which breaks `CreateProcess` on Windows.

## What Is Here

- `docs/KNOWN_GOOD_ENV.md`: exact versions that worked
- `docs/REBUILD_STEPS.md`: repeatable rebuild instructions
- `scripts/build_flashattn_windows.cmd`: wrapper to compile from source
- `scripts/collect_env.py`: emits a JSON fingerprint of the local machine, Python env, and toolchain
- `scripts/patch_torch_cpp_extension_windows.py`: patches the local torch env to emit a Windows-safe `nvcc` launcher
- `scripts/patch_flash_attn_setup.py`: patches `setup.py` in the source tree for explicit `BUILD_TARGET` handling
- `scripts/patch_flash_attn_sm120_backward.py`: patches the local upstream FA4 source tree so the SM120 backward path does not reference `dQ_single_wg` before assignment
- `scripts/smoke_test_flash_attn.py`: import and CUDA execution smoke test
- `scripts/test_fa4_windows_shim.py`: probes the repo-local FA4 import shim and can run a tiny CUDA forward smoke test
- `scripts/validate_fa4_windows_shim.py`: broader validation matrix for the Windows FA4 shim
- `scripts/probe_cutlass_runtime.py`: reports which CUTLASS runtime the native FA4 probe is actually using and why
- `scripts/probe_cutlass_cubin_loader.py`: compiles a tiny NVRTC kernel and verifies the Windows CUTLASS runtime shim can load and resolve it
- `scripts/probe_native_fa4_combine.py`: isolated native-path probe for the upstream forward-combine compile family
- `scripts/probe_native_fa4_block_sparsity.py`: isolated native-path probe for the upstream block-sparsity compile family
- `scripts/probe_native_fa4_import.py`: isolated native-path import probe using compatibility shims instead of the stable fallback
- `scripts/probe_native_fa4_forward.py`: native-path forward probe covering base dense attention, dense `softcap`, dense `mask_mod`, and varlen `softcap`
- `scripts/probe_native_fa4_backward.py`: dense + varlen native-path backward parity probe against the stable Windows shim
- `runtime_compat/`: installable Windows import-compat package that restores the old top-level `cuda` API shape and provides a discoverable `nvidia_cutlass_dsl` module name
- `cutlass_runtime/`: installable top-level `cutlass` wrapper that promotes the repo's native probe package into a normal Windows import surface
- `shims/`: repo-local compatibility shims used only for FA4 Windows probing
- `native_probe_shims/`: isolated import/runtime scaffolding that pushes the native FA4 path farther without touching the stable fallback
- `patches/*.patch`: reference diffs for the required source edits

## What This Does Not Do

- It does not automatically speed up Ollama or Modelfile-based inference by itself.
- It does not claim FlashAttention 4 is working on this exact stack yet.
- It does not bundle the upstream FlashAttention source tree.

## FA4 Status

FlashAttention 4 was tested in a dedicated `.venv_fa4` environment on the same machine.

Current status:

- `flash-attn-4` can be installed as an editable package from the local `flash_attn/cute` tree
- after installing `runtime_compat/` and `cutlass_runtime/`, a clean top-level `import flash_attn.cute` now works without manually prepending `native_probe_shims` to `sys.path`
- a repo-local shim package under `shims/` now shadows `flash_attn.cute` in `.venv_fa4`
- the shimmed dense `flash_attn_func` forward and backward probes both match torch SDPA exactly in the isolated test
- the shimmed varlen path was also cross-checked against FA2 behavior for unequal sequence lengths and matched closely
- the shim now also supports `learnable_sink`
- the shim now supports dense `mask_mod` fallbacks and varlen `score_mod` fallbacks, including global offset-aware `seqlen_info.offset_q` / `offset_k`
- custom-mod validation now also covers backward parity for dense `mask_mod` and varlen `score_mod`
- varlen validation also covers the aux-only `score_mod(..., aux_tensors)` callable form
- validation now also covers dense and varlen `softcap`
- validation now also covers varlen `seqused_k` truncation
- validation now also covers varlen `seqused_q` plus the mixed packed/padded layout paths
- validation now also covers `info`-named seqlen modifier signatures in addition to `seqlen_info`
- validation now also covers dynamic split-aware forward-combine masking for both batched and varlen layouts
- validation now also covers block-sparse forward parity against the dense mask-mod reference path
- the shim passes the broader validation matrix in `scripts/validate_fa4_windows_shim.py`
- the native probe now auto-prefers a real modern CUTLASS package if one ever becomes importable in this env
- the isolated native probe path now imports `flash_attn.cute` successfully with the runtime-owned `cutlass_runtime/` package and currently reports `cutlass_shim_module_count=0`
- the native probe now replaces recognized FA4 forward-kernel `cute.compile(...)` calls with a real Windows bridge object
- the native probe now also replaces recognized FA4 backward preprocess, main backward, and backward postprocess `cute.compile(...)` calls with bridge objects
- the tiny native-path CUDA forward probe now reaches numerically sane dense output through that bridge, with close parity versus SDPA even when LSE is requested
- the new native-path backward probe now reaches dense and varlen backward parity against the stable Windows shim with `0.0` output and grad diffs in the seeded checks
- the upstream forward-combine compile family now also resolves through a real `NativeProbeForwardCombineBridge`, with exact batched and varlen parity versus the stable Windows shim in the new combine probe
- the forward-combine bridge now also handles `num_splits_dynamic_ptr`, with exact batched and varlen parity versus the stable Windows shim in the dynamic-split combine probe cases
- the forward-combine bridge is now backed by a repo-built Windows extension module (`fa4_windows_native_combine_ext.cp313-win_amd64.pyd`) instead of the pure Python shim core
- the upstream `compute_block_sparsity(...)` compile family now also resolves through a real `NativeProbeBlockSparsityBridge`, with exact parity versus the stable Windows shim for both exact and fast-sampling test cases
- the forward bridge now also supports end-to-end block-sparse execution onto the stable Windows shim path
- the upstream public SM120 `flash_attn_func(...)` block-sparse wrapper path now also reaches that bridge cleanly, with exact forward and backward parity versus the stable Windows shim in the public probe cases
- the backward bridge now also preserves forward-only feature metadata so unsupported SM120 backward surfaces can fall back compatibly onto the stable Windows shim without changing the user-facing API call shape
- widened native-path modifier probes now show:
  - dense `softcap` forward and backward parity are exact against the stable Windows shim
  - dense `learnable_sink` forward and backward parity are exact against the stable Windows shim
  - dense `mask_mod` forward and backward parity are exact against the stable Windows shim
  - dense block-sparse forward and backward parity are exact against the stable Windows shim through the public SM120 wrapper path
  - varlen `softcap` forward parity is exact against the stable Windows shim
  - varlen `seqused_q` / `seqused_k` forward and backward parity are exact against the stable Windows shim
  - varlen `score_mod` forward parity is exact against the stable Windows shim
  - varlen `seqused_q` / `seqused_k` plus `score_mod` backward parity is exact against the stable Windows shim
- the compiled cache entry for that path is now `NativeProbeForwardBridge`, not a dead placeholder
- the CUTLASS runtime probe now shows that raw `cutlass`, raw `cuda`, and raw `nvidia_cutlass_dsl` imports all succeed from repo-local Windows wrapper packages
- the native probe cubin loader now also succeeds end-to-end through `cutlass_runtime/src/cutlass/base_dsl/runtime/cuda.py` using `cudaLibraryLoadData`, `cudaLibraryGetKernel`, and `cudaLibraryUnload`
- a small installable compat package now exists under `runtime_compat/`; once installed into `.venv_fa4`, the raw Windows import surface improves to:
  - `raw_cutlass_import=ok`
  - `raw_cuda_import=ok` from `runtime_compat/src/cuda/__init__.py`
  - `raw_nvidia_cutlass_dsl_import=ok` from `runtime_compat/src/nvidia_cutlass_dsl/__init__.py`
  - then, after also installing `cutlass_runtime/`, `raw_cutlass_spec` resolves to `cutlass_runtime/src/cutlass/__init__.py` and clean `import flash_attn.cute` succeeds without manual probe-path injection
  - the remaining blocker then shrinks to the full native runtime gap: the native probe no longer leaks active `cutlass.*` modules through `native_probe_shims`, but it still relies on repo-local bridge objects instead of a true native CuTe/CUTLASS DSL backend

Current native-probe import mode:

- the probe scripts now import `cutlass` through the installable `cutlass_runtime/` wrapper first
- the probe scripts now also import `flash_attn.cute` through the installable `flash_attn_runtime/` overlay first
- current observed probe mode is `runtime-local-core`
- that means the Windows import surface is now exercising the repo's runtime-owned `cutlass` package and repo-owned `flash_attn.cute.interface` overlay directly; the remaining blocker is no longer legacy-root delegation, but the lack of a standalone native CuTe/CUTLASS DSL compiler/runtime behind `cutlass.cute.compile`

The root native blocker is now pinned down more precisely than just "missing wheels". In `.venv_fa4`, the editable CUTLASS Python tree is present, but the newer FA4 stack expects a much larger CUTLASS DSL surface than that tree provides. The repo-local `cutlass_runtime/` package now owns the active `cutlass`, `cutlass.cute`, `cutlass.cutlass_dsl`, `cutlass.pipeline`, `cutlass.utils`, and `cutlass._mlir` import surfaces for the current native probe path, while `runtime_compat/` fixes the raw CUDA / `nvidia_cutlass_dsl` import mismatch directly and `flash_attn_runtime/` owns the active `flash_attn.cute` overlay. Recognized FA4 `cute.compile(...)` calls still resolve to bridge objects, but the forward-combine family is now a real compiled Windows backend slice through `fa4_windows_native_combine_ext`, and the remaining bridge families still route onto the validated Windows shim path with exact or near-exact parity in the probes. That means the remaining blocker is no longer package discovery, a dead cubin hook, active `cutlass.*` leakage from `native_probe_shims`, or a live runtime dependency on an external `flash_attn.cute.interface` file. It is the absence of a full native CuTe DSL compiler/runtime path behind the rest of `cutlass.cute.compile` on this Windows stack. The current stable path is now a hybrid: one compiled Windows backend slice for forward-combine plus selective bridge objects for the rest of the FA4 surface.

Current checkpoint caveat:

- the repo-owned overlay at `flash_attn_runtime/src/flash_attn/cute/interface.py` now carries the active runtime interface path
- the sync helper at `scripts/sync_flash_attn_runtime_overlay.py` refreshes that overlay from the upstream clone and reapplies the SM120 compatibility patch when needed
- the public backward path now cleanly accepts dense `deterministic=True`, plain varlen `score_mod`, varlen `seqused + score_mod`, and varlen `softcap + score_mod`, all of which probe exact against the stable Windows shim
- the first compiled backend slice now lives in `cutlass_runtime/src/cutlass/cute/_native_backend.py` and `cutlass_runtime/src/cutlass/cute/_native_combine_backend.cpp`, with `scripts/build_native_combine_backend.py` available to rebuild the `.pyd`

See `docs/FA4_WINDOWS_STATUS.md` for the exact attempted install path and blocker.

## Quick Start

1. Clone this repo.
2. Clone `https://github.com/sdbds/flash-attention-for-windows` into `third_party/flash-attention-for-windows`.
3. Run `python scripts/collect_env.py --json-out local_env.json` if you want a one-shot machine fingerprint before touching the build.
4. Follow `docs/REBUILD_STEPS.md`.
5. Compare your local machine against `docs/KNOWN_GOOD_ENV.md` before debugging build failures.

## License

- MIT for the helper scripts and docs in this repo.
- Upstream FlashAttention code stays under its own upstream license.
