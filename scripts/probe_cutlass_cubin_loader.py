"""Probe the Windows CUTLASS runtime cubin loader with a tiny NVRTC kernel."""

from __future__ import annotations

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
import torch

import cutlass


_KERNEL_SRC = r"""
extern "C" __global__ void fa4_probe_add_one(float* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] += 1.0f;
}
"""


def _check_nvrtc(err, opname: str) -> None:
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"{opname} failed with NVRTC error: {err}")


def _device_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _compile_probe_kernel() -> bytes:
    err, program = nvrtc.nvrtcCreateProgram(
        _KERNEL_SRC.encode("utf-8"),
        b"fa4_probe_add_one.cu",
        0,
        [],
        [],
    )
    _check_nvrtc(err, "nvrtcCreateProgram")

    options = [
        f"-arch=sm_{_device_arch()}".encode("utf-8"),
        b"--std=c++17",
    ]
    err, = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_err, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        _check_nvrtc(log_err, "nvrtcGetProgramLogSize")
        log = b" " * log_size
        log_err, = nvrtc.nvrtcGetProgramLog(program, log)
        _check_nvrtc(log_err, "nvrtcGetProgramLog")
        raise RuntimeError(f"nvrtcCompileProgram failed: {log.decode(errors='replace')}")

    err, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
    _check_nvrtc(err, "nvrtcGetCUBINSize")
    cubin = b" " * cubin_size
    err, = nvrtc.nvrtcGetCUBIN(program, cubin)
    _check_nvrtc(err, "nvrtcGetCUBIN")
    return cubin


def main() -> int:
    runtime_cuda = cutlass.base_dsl.runtime.cuda
    cubin = _compile_probe_kernel()
    library = runtime_cuda.load_cubin_module_data(cubin)
    kernel = runtime_cuda.get_module_function(library, "fa4_probe_add_one")

    print(f"cutlass_spec={getattr(getattr(cutlass, '__spec__', None), 'origin', '<missing>')}")
    print(f"cutlass_file={getattr(cutlass, '__file__', '<missing>')}")
    print(f"probe_mode={getattr(cutlass, 'NATIVE_PROBE_MODE', '<missing>')}")
    print(f"runtime_cuda_module={getattr(runtime_cuda, '__file__', '<missing>')}")
    print(f"cubin_size={len(cubin)}")
    print(f"library_type={type(library).__name__}")
    print(f"kernel_type={type(kernel).__name__}")

    unload = getattr(runtime_cuda, "unload_module", None)
    if callable(unload):
        unload(library)
        print("library_unload=ok")
    else:
        print("library_unload=<missing>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
