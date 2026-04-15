# FA4 Windows Worklog

## 2026-04-15

### What moved

- Extended the stable Windows shim in `shims/flash_attn/cute/__init__.py` to cover:
  - paged-KV materialization for varlen calls
  - shared varlen KV preparation/finalization helpers
  - repo-local varlen block-sparse forward replay
  - combined `softcap + score_mod` replay on the shim path
- Extended the native probe bridge in `native_probe_shims/cutlass/cute/_compile_bridge.py` to cover:
  - extraction of combined `softcap + score_mod` closures
  - varlen block-sparse forward bridge replay
  - repo-local `compat_replay_varlen_backward(...)` for Windows fallback backward replay
  - optional block-sparse backward replay through that same helper
- Expanded the upstream patch layer in `scripts/patch_flash_attn_sm120_backward.py` so the patched FA4 wrapper now:
  - composes `softcap` with `score_mod`
  - removes the SM120 paged-KV forward guard
  - removes the varlen mask-mod forward guard
  - removes the varlen block-sparsity forward guard
  - saves varlen compat metadata on the autograd context
  - reroutes paged varlen backward through the repo-local shim replay helper
- Expanded the native probes:
  - `scripts/probe_native_fa4_forward.py`
  - `scripts/probe_native_fa4_backward.py`
  - new coverage now includes varlen paged-KV, varlen `softcap + score_mod`, and internal varlen block-sparse replay

### Verification run

- `.\.venv_fa4\Scripts\python.exe scripts\probe_native_fa4_forward.py`
  - `varlen_paged_kv_out_max_diff=0.0`
  - `varlen_paged_kv_lse_max_diff=0.0`
  - `varlen_softcap_score_mod_out_max_diff=0.0`
  - `varlen_softcap_score_mod_lse_max_diff=0.0`
  - `varlen_block_sparse_internal_out_max_diff=0.0`
  - `varlen_block_sparse_internal_lse_max_diff=0.0`
- `.\.venv_fa4\Scripts\python.exe scripts\probe_native_fa4_backward.py`
  - `varlen_paged_kv_grad_max_diff=0.0`
  - `varlen_softcap_score_mod_grad_max_diff=0.0`
  - `varlen_block_sparse_internal_grad_max_diff=0.0`
- `.\.venv_fa4\Scripts\python.exe scripts\validate_fa4_windows_shim.py`
  - `validation=ok`

### What is still honestly missing

- The environment is still not a true Windows-native CuTe/CUTLASS DSL runtime.
- The current probe mode is still `runtime-wrapper+legacy-core`.
- The repo-local compat and wrapper layers are doing real work, but they still delegate into the legacy editable CUTLASS tree instead of a standalone modern Windows runtime package.
- The live upstream file being patched is outside this repo tree:
  - `..\third_party\flash-attention-for-windows\flash_attn\cute\interface.py`
  - the reproducible source of truth for that edit remains `scripts/patch_flash_attn_sm120_backward.py`

### Next sensible targets

- Fold the latest upstream-interface compat edits fully into the patch script so the external file can be regenerated cleanly from a fresh checkout.
- Add shim validator coverage for paged-KV and combined `softcap + score_mod` so those paths are checked outside the native-probe scripts too.
- Keep pushing the real blocker:
  - replacing `runtime-wrapper+legacy-core` with a genuine Windows CuTe/CUTLASS DSL runtime/compiler path.
