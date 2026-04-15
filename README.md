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
- `scripts/smoke_test_flash_attn.py`: import and CUDA execution smoke test
- `scripts/test_fa4_windows_shim.py`: probes the repo-local FA4 import shim and can run a tiny CUDA forward smoke test
- `scripts/validate_fa4_windows_shim.py`: broader validation matrix for the Windows FA4 shim
- `scripts/probe_cutlass_runtime.py`: reports which CUTLASS runtime the native FA4 probe is actually using and why
- `scripts/probe_native_fa4_import.py`: isolated native-path import probe using compatibility shims instead of the stable fallback
- `scripts/probe_native_fa4_forward.py`: tiny native-path CUDA forward probe plus SDPA sanity check
- `scripts/probe_native_fa4_backward.py`: dense + varlen native-path backward parity probe against the stable Windows shim
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
- the native Windows import path still does not work without extra shims
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
- the shim passes the broader validation matrix in `scripts/validate_fa4_windows_shim.py`
- the native probe now auto-prefers a real modern CUTLASS package if one ever becomes importable in this env
- the isolated native probe path now imports `flash_attn.cute` successfully under `native_probe_shims/`
- the native probe now replaces recognized FA4 forward-kernel `cute.compile(...)` calls with a real Windows bridge object
- the native probe now also replaces recognized FA4 backward preprocess, main backward, and backward postprocess `cute.compile(...)` calls with bridge objects
- the tiny native-path CUDA forward probe now reaches numerically sane dense output through that bridge, with close parity versus SDPA even when LSE is requested
- the new native-path backward probe now reaches dense and varlen backward parity against the stable Windows shim with `0.0` output and grad diffs in the seeded checks
- the compiled cache entry for that path is now `NativeProbeForwardBridge`, not a dead placeholder
- the CUTLASS runtime probe currently shows `nvidia-cutlass-dsl` metadata is installed, but no separate modern CUTLASS runtime package is importable on this Windows env, so the probe still falls back to the legacy editable CUTLASS tree plus compatibility shims

The root native blocker is now pinned down more precisely than just "missing wheels". In `.venv_fa4`, the editable CUTLASS Python tree is present, but the newer FA4 stack expects a much larger CUTLASS DSL surface than that tree provides. The isolated `native_probe_shims/` layer now bridges enough of the CUDA API shape, CUTLASS package shape, CuTe module tree, MLIR dialects, pipeline classes, and Unix-only `fcntl` import to make the real `flash_attn.cute` import succeed. It also now intercepts recognized FA4 forward-kernel `cute.compile(...)` calls and returns a real `NativeProbeForwardBridge` object that routes execution onto the validated Windows shim path, producing numerically sane dense outputs and LSE tensors with close parity versus SDPA in the probe. The new runtime probe still shows why this is not native CuTe codegen yet: `nvidia-cutlass-dsl` is installed only as the top-level distribution metadata here, while the separate CUTLASS DSL runtime packages FA4 expects are not importable for this Windows env, so the probe must still fall back to the legacy editable CUTLASS tree plus compatibility shims. The current stable path is therefore still the Windows compatibility shim and selective bridge objects, not a true native FA4 kernel path.

Current checkpoint caveat:

- the backward checkpoint also depends on a small SM120-side fix in the local upstream source tree at `third_party/flash-attention-for-windows/flash_attn/cute/interface.py`, where `dQ_single_wg` now defaults to `False` before the arch split so the SM120 path does not reference it before assignment

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
