"""Build the compiled dense backward backend for the native FA4 probe path."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "cutlass_runtime" / "src"))

    from cutlass.cute._native_dense_bwd_backend import (
        load_native_dense_bwd_backend,
        native_dense_bwd_backend_status,
    )

    module = load_native_dense_bwd_backend(verbose=True, force_rebuild=False)
    status = native_dense_bwd_backend_status()
    print(f"native_dense_bwd_loaded={module is not None}")
    print(f"native_dense_bwd_module_file={status['module_file']}")
    print(f"native_dense_bwd_source={status['source']}")
    print(f"native_dense_bwd_build_dir={status['build_dir']}")
    print(f"native_dense_bwd_error={status['error']}")
    return 0 if module is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
