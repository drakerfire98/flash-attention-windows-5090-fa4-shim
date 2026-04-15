"""Runtime-local CUDA binary-loader helpers for the Windows CUTLASS probe path."""

from __future__ import annotations

from collections.abc import Buffer

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart


def _normalize_blob(blob: bytes | bytearray | memoryview | Buffer) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    if isinstance(blob, memoryview):
        return blob.tobytes()
    return bytes(blob)


def _check_driver(err, opname: str) -> None:
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{opname} failed with CUDA driver error: {err}")


def _check_runtime(err, opname: str) -> None:
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{opname} failed with CUDA runtime error: {err}")


def initialize_cuda() -> None:
    (err,) = cuda.cuInit(0)
    _check_driver(err, "cuInit")


def ensure_current_context(device: int | None = None) -> None:
    initialize_cuda()
    err, ctx = cuda.cuCtxGetCurrent()
    if err == cuda.CUresult.CUDA_SUCCESS and ctx not in (None, 0):
        return

    if device is None:
        try:
            import torch

            device = int(torch.cuda.current_device())
        except Exception:
            device = 0

    (err,) = cudart.cudaSetDevice(int(device))
    _check_runtime(err, "cudaSetDevice")
    (err,) = cudart.cudaFree(0)
    _check_runtime(err, "cudaFree")

    err, ctx = cuda.cuCtxGetCurrent()
    _check_driver(err, "cuCtxGetCurrent")
    if ctx in (None, 0):
        raise RuntimeError("CUDA context bootstrap succeeded but cuCtxGetCurrent returned no active context")


def load_cubin_module_data(cubin_data: bytes | bytearray | memoryview | Buffer):
    ensure_current_context()
    blob = _normalize_blob(cubin_data)
    err, library = cudart.cudaLibraryLoadData(blob, None, None, 0, None, None, 0)
    _check_runtime(err, "cudaLibraryLoadData")
    return library


def load_module_data(blob: bytes | bytearray | memoryview | Buffer):
    return load_cubin_module_data(blob)


def get_module_function(module, function_name: str):
    err, kernel = cudart.cudaLibraryGetKernel(module, function_name.encode("utf-8"))
    _check_runtime(err, "cudaLibraryGetKernel")
    return kernel


def unload_module(module) -> None:
    (err,) = cudart.cudaLibraryUnload(module)
    _check_runtime(err, "cudaLibraryUnload")
