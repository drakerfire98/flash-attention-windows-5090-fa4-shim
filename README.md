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
- `shims/`: repo-local compatibility shims used only for FA4 Windows probing
- `patches/*.patch`: reference diffs for the two required source edits

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

The root native blocker is now pinned down more precisely than just "missing wheels". In `.venv_fa4`, the editable CUTLASS Python tree is present, but a plain import still fails because CUTLASS expects the older top-level CUDA Python API shape (`from cuda import __version__, cuda, cudart, nvrtc`) while the installed `cuda-python==13.2.0` exposes modules under `cuda.bindings.*`. Even after an in-memory compatibility patch makes `import cutlass` succeed, the linked tree still does not contain `cutlass.cute`, which is the module `flash_attn.cute` needs. The current stable path is therefore a Windows compatibility shim, not a native FA4 kernel path.

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
