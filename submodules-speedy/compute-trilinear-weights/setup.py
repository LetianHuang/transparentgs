from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name='compute_trilinear_weights',
    ext_modules=[
        CUDAExtension(
            name='compute_trilinear_weights._C',
            sources=[
                "trilinear_weights.cu"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)