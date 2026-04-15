from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent


setup(
    name="fa4_windows_native_combine_ext",
    ext_modules=[
        CppExtension(
            name="fa4_windows_native_combine_ext",
            sources=[str(HERE / "_native_combine_backend.cpp")],
            extra_compile_args=["/O2", "/std:c++17", "/DNOMINMAX"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
