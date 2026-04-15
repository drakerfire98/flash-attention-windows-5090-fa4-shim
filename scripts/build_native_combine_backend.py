"""Build the compiled Windows combine backend for the native FA4 runtime path."""

from __future__ import annotations

from _native_probe_setup import install_native_probe_paths


def main() -> int:
    install_native_probe_paths()

    from cutlass.cute._native_backend import (
        load_native_combine_backend,
        native_combine_backend_status,
    )

    module = load_native_combine_backend(verbose=True, force_rebuild=False)
    status = native_combine_backend_status()
    print(f"native_combine_loaded={module is not None}")
    print(f"native_combine_name={status['name']}")
    print(f"native_combine_source={status['source']}")
    print(f"native_combine_build_dir={status['build_dir']}")
    print(f"native_combine_module_file={status['module_file']}")
    print(f"native_combine_error={status['error']}")
    return 0 if module is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
