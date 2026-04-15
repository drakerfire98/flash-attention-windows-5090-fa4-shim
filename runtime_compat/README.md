# fa4-windows-runtime-compat

Small Windows compatibility package for the FlashAttention 4 / CUTLASS probing
environment in this repo.

What it does:

- restores the older top-level `cuda` Python API shape expected by the legacy
  CUTLASS Python tree:
  - `from cuda import __version__, cuda, cudart, nvrtc`
- provides an importable `nvidia_cutlass_dsl` module name on Windows, where the
  published distribution metadata may exist without a matching importable module

What it does not do:

- it does not provide native CuTe code generation
- it does not replace the validated fallback under `shims/`
- it does not make the legacy editable CUTLASS tree become a true modern
  `cutlass.cute` runtime by itself

Install into the isolated FA4 env:

```powershell
.\.venv_fa4\Scripts\python.exe -m pip install -e .\runtime_compat
```
