# FA4 Windows Worklog

## 2026-04-15

### What moved

- Replaced `cutlass_runtime/src/cutlass/__init__.py` with a repo-local root implementation:
  - `cutlass.__file__` now resolves to the local runtime package instead of the legacy editable CUTLASS root
  - the package now reports `NATIVE_PROBE_MODE=runtime-local-core`
  - it eagerly exposes local / repo-local `base_dsl`, `pipeline`, `utils`, `cutlass_dsl`, and `_mlir`
- Added a repo-local `cutlass.cute` package under `cutlass_runtime/src/cutlass/cute/`:
  - `cutlass.cute.__file__` now resolves to the local runtime package instead of `native_probe_shims/`
  - local runtime-owned modules now cover `__init__.py`, `_pycute_compat.py`, `runtime.py`, `tensor.py`, `_tvm_ffi_args_spec_converter.py`, and the `nvgpu/` placeholder tree
  - the runtime-owned `_compile_bridge.py` now contains the heavy bridge logic directly
  - `native_probe_shims/cutlass/cute/_compile_bridge.py` is now just a compatibility wrapper back to the runtime-owned bridge
- Added a repo-local `cutlass.base_dsl` package under `cutlass_runtime/src/cutlass/base_dsl/`:
  - `cutlass.base_dsl.__file__` and `cutlass.base_dsl.runtime.cuda.__file__` now resolve to the runtime package
  - local runtime-owned modules now cover `__init__.py`, `arch.py`, `typing.py`, `tvm_ffi_builder.py`, `runtime/__init__.py`, and `runtime/cuda.py`
  - `native_probe_shims/cutlass/base_dsl/runtime/cuda.py` is now just a compatibility wrapper back to the runtime-owned CUDA loader helpers
- Added repo-local runtime-owned compatibility modules for the remaining CUTLASS import surfaces that were still leaking through `native_probe_shims/`:
  - `cutlass_runtime/src/cutlass/cutlass_dsl.py`
  - `cutlass_runtime/src/cutlass/pipeline.py`
  - `cutlass_runtime/src/cutlass/utils/*`
  - `cutlass_runtime/src/cutlass/_mlir/*`
  - on the current native probe import path, `cutlass_shim_module_count=0`
- Replaced the installable wrapper modules in:
  - `cutlass_runtime/src/_probe_helpers.py`
  - `cutlass_runtime/src/fcntl.py`
  - these no longer `exec(...)` the files under `native_probe_shims/`
- Added a local CuTe / PyCuTe compatibility layer in:
  - `native_probe_shims/cutlass/cute/_pycute_compat.py`
  - `native_probe_shims/cutlass/cute/__init__.py` now imports that local compatibility layer instead of `from pycute import *`
  - importing `cutlass.cute` through the real FA4 probe path no longer loads the external legacy `pycute` package
- Replaced `native_probe_shims/cutlass/utils/__init__.py` with a local minimal utils compatibility package so `quack` and the FA4 modules stop depending on the legacy CUTLASS utils root
- Extended the stable Windows shim in `shims/flash_attn/cute/__init__.py` to cover:
  - paged-KV materialization for varlen calls
  - shared varlen KV preparation/finalization helpers
  - repo-local varlen block-sparse forward replay
  - combined `softcap + score_mod` replay on the shim path
- The heavy native probe bridge logic is now owned by `cutlass_runtime/src/cutlass/cute/_compile_bridge.py`:
  - extraction of combined `softcap + score_mod` closures
  - varlen block-sparse forward bridge replay
  - repo-local `compat_replay_varlen_backward(...)` for Windows fallback backward replay
  - optional block-sparse backward replay through that same helper
  - `native_probe_shims/cutlass/cute/_compile_bridge.py` now only wraps that runtime-owned implementation for compatibility
- Expanded the upstream patch layer in `scripts/patch_flash_attn_sm120_backward.py` so the patched FA4 wrapper now:
  - composes `softcap` with `score_mod`
  - removes the SM120 paged-KV forward guard
  - removes the varlen mask-mod forward guard
  - removes the varlen block-sparsity forward guard
  - saves varlen compat metadata on the autograd context
  - reroutes paged varlen backward through the repo-local shim replay helper
  - verifies its expected patch markers and fails if forbidden upstream guard fragments reappear
- Expanded the native probes:
  - `scripts/probe_native_fa4_forward.py`
  - `scripts/probe_native_fa4_backward.py`
  - new coverage now includes varlen paged-KV, varlen `softcap + score_mod`, and internal varlen block-sparse replay
- Added a second compiled Windows backend slice for the plain dense forward family:
  - `cutlass_runtime/src/cutlass/cute/_native_dense_backend.py`
  - `cutlass_runtime/src/cutlass/cute/_native_dense_backend.cpp`
  - `cutlass_runtime/src/cutlass/cute/_native_dense_setup.py`
  - `scripts/build_native_dense_backend.py`
  - the native forward bridge now prefers that backend for the no-window, no-modifier dense family
  - the native backward replay path now also reuses that compiled dense backend for the same plain family before falling back to the validated shim
- Relaxed the last hard SplitKV block-sparsity failure in the repo-local overlay:
  - `flash_attn_runtime/src/flash_attn/cute/interface.py` now degrades block sparsity with `num_splits > 1` to a compatible non-split path instead of raising `NotImplementedError`
  - the same overlay file now updates the old pack-GQA TODO comments to explicit compatibility-fallback comments
- Added shared native probe setup in `scripts/_native_probe_setup.py`:
  - native probe scripts now install the runtime/shim paths consistently
  - native probe scripts now auto-ensure the external FA4 patch before import or execution
  - the probe flow no longer relies on remembering to run the patch step manually first
- Expanded `scripts/validate_fa4_windows_shim.py`:
  - added broader validator coverage for varlen paged-KV forward/backward parity
  - added broader validator coverage for combined varlen `softcap + score_mod` forward/backward parity

### Verification run

- `.\.venv_fa4\Scripts\python.exe scripts\probe_cutlass_runtime.py`
  - `raw_cutlass_spec` now resolves to `cutlass_runtime/src/cutlass/__init__.py`
  - `raw_cuda_spec` now resolves to `runtime_compat/src/cuda/__init__.py`
  - `cutlass_probe_mode=runtime-local-core`
  - `cute_file` now resolves to `cutlass_runtime/src/cutlass/cute/__init__.py`
  - `cute_compile_bridge_file` now resolves to `cutlass_runtime/src/cutlass/cute/_compile_bridge.py`
  - `runtime_cuda_file` now resolves to `cutlass_runtime/src/cutlass/base_dsl/runtime/cuda.py`
  - `pycute_loaded=False`
  - `cutlass_shim_module_count=0`
  - `modern_runtime_ready=True`
- `.\.venv_fa4\Scripts\python.exe scripts\probe_native_fa4_import.py`
  - `native_import=ok`
  - `cutlass_cute_file` resolves to `cutlass_runtime/src/cutlass/cute/__init__.py`
  - `pycute_loaded=False`
  - `cutlass_shim_module_count=0`
- `.\.venv_fa4\Scripts\python.exe scripts\patch_flash_attn_sm120_backward.py ..\third_party\flash-attention-for-windows\flash_attn\cute\interface.py`
  - `already patched`
  - `verification=ok`
- `.\.venv_fa4\Scripts\python.exe scripts\probe_native_fa4_forward.py`
  - `patched_interface=..\third_party\flash-attention-for-windows\flash_attn\cute\interface.py`
  - `varlen_paged_kv_out_max_diff=0.0`
  - `varlen_paged_kv_lse_max_diff=0.0`
  - `varlen_softcap_score_mod_out_max_diff=0.0`
  - `varlen_softcap_score_mod_lse_max_diff=0.0`
  - `varlen_block_sparse_internal_out_max_diff=0.0`
  - `varlen_block_sparse_internal_lse_max_diff=0.0`
  - `compiled_repr_sample=<NativeProbeForwardBridge FlashAttentionForwardSm120 dense_backend=compiled>`
  - `native_dense_backend_post["loaded"] = True`
- `.\.venv_fa4\Scripts\python.exe scripts\probe_native_fa4_backward.py`
  - `patched_interface=..\third_party\flash-attention-for-windows\flash_attn\cute\interface.py`
  - `varlen_paged_kv_grad_max_diff=0.0`
  - `varlen_softcap_score_mod_grad_max_diff=0.0`
  - `varlen_block_sparse_internal_grad_max_diff=0.0`
  - `native_dense_backend_post["loaded"] = True`
- `.\.venv_fa4\Scripts\python.exe scripts\build_native_dense_backend.py`
  - `native_dense_loaded=True`
  - `native_dense_error=None`
- inline SplitKV block-sparsity smoke check via `.venv_fa4`
  - `splitkv_block_sparse_out_finite=True`
  - `splitkv_block_sparse_lse_finite=True`
- `.\.venv_fa4\Scripts\python.exe scripts\probe_native_fa4_combine.py`
  - `patched_interface=..\third_party\flash-attention-for-windows\flash_attn\cute\interface.py`
  - all tested combine parity cases remain exact
- `.\.venv_fa4\Scripts\python.exe scripts\validate_fa4_windows_shim.py`
  - `validation=ok`
  - `varlen_paged_kv_out_max_diff=0.0`
  - `varlen_paged_kv_dq_max_diff=0.0`
  - `varlen_softcap_score_mod_out_max_diff=0.0`
  - `varlen_softcap_score_mod_dq_max_diff=0.0`

### What is still honestly missing

- The environment is still not a true Windows-native CuTe/CUTLASS DSL runtime.
- The root package, the top-level `cutlass.cute` package, the heavy compile bridge, the `base_dsl.runtime.cuda` loader path, and the currently imported `cutlass_dsl` / `pipeline` / `utils` / `_mlir` surfaces are now local.
- The remaining blocker is no longer active `cutlass.*` leakage from `native_probe_shims`; the forward-combine family and the plain dense forward family now build through real compiled Windows extensions, but the rest of `cutlass.cute.compile` still resolves recognized kernels to repo-local bridge objects instead of a true compiled Windows CuTe/CUTLASS DSL backend.
- The current probe mode is now `runtime-local-core`, which is better than `runtime-wrapper+legacy-core` but still not a true native compiler/runtime.
- The active `flash_attn.cute.interface` surface is now repo-local under `flash_attn_runtime/src/flash_attn/cute/interface.py`; the upstream clone is now a refresh source, not a live runtime dependency.
- The public backward path now accepts dense `deterministic=True`, plain varlen `score_mod`, varlen `seqused + score_mod`, and varlen `softcap + score_mod` through the replay bridge, and the native backward probe reports exact parity for those cases.
- The forward-combine probe now reports `NativeCompiledForwardCombineBridge` with `backend=compiled`, backed by `fa4_windows_native_combine_ext.cp313-win_amd64.pyd`, and all tested batched/varlen/dynamic combine cases remain exact.
- The forward probe now reports `NativeProbeForwardBridge ... dense_backend=compiled`, backed by `fa4_windows_native_dense_ext.cp313-win_amd64.pyd`, and the backward replay probe reuses that same dense backend without regressing the seeded parity checks.

### Next sensible targets

- Keep `scripts/sync_flash_attn_runtime_overlay.py` and `scripts/patch_flash_attn_sm120_backward.py` as the single source of truth for overlay refreshes after each upstream sync.
- Extend validator/probe coverage only when new FA4 surface area is actually added.
- Keep pushing the real blocker:
  - replacing more of `runtime-local-core` with genuine compiled Windows backend slices instead of repo-local bridge objects.
  - the next high-value targets are backward postprocess/preprocess helpers and a broader native forward family beyond the plain dense no-modifier path.
