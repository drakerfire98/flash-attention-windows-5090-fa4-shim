# fa4-windows-cutlass-runtime

Small editable wrapper that exposes the repo's Windows native-probe CUTLASS
surface as a real top-level `cutlass` package in `.venv_fa4`.

Purpose:

- make `import cutlass` resolve to the repo's Windows probe runtime instead of
  the legacy editable CUTLASS tree alone
- make `import flash_attn.cute` advance without manual `sys.path.insert(...)`
  calls for `native_probe_shims`
- keep the actual probe implementation in one place under `native_probe_shims/`

Install:

```powershell
.\.venv_fa4\Scripts\python.exe -m pip install -e .\cutlass_runtime
```

This package does not provide a real CuTe compiler yet. It only promotes the
existing Windows probe surface into a normal top-level import path.
