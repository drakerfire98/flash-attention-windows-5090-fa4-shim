from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent


setup(
    name="fa4_win_vbwd_ext",
    ext_modules=[
        CppExtension(
            name="fa4_win_vbwd_ext",
            sources=[str(HERE / "_native_varlen_bwd_backend.cpp")],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
