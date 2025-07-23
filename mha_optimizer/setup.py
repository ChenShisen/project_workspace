import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- 1. 获取ABI兼容性标志 ---
CXX_FLAGS = ['-O3', '-std=c++17']
if torch.compiled_with_cxx11_abi():
    CXX_FLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=1')
else:
    CXX_FLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=0')

# --- 2. 只为你的GPU架构编译 ---
ARCH_FLAGS = ['-gencode', 'arch=compute_80,code=sm_80']

setup(
    name='mha_optimizer_cpp',
    version='1.1.0', # Stable No-CUTLASS version
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'mha_optimizer_cpp',
            [
                'csrc/mha_optimizer.cpp',
                'csrc/packed_attention_kernel.cu',
            ],
            # !! 关键：移除所有对CUTLASS的依赖 !!
            # include_dirs=[],
            extra_compile_args={
                'cxx': CXX_FLAGS,
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '-Xcompiler', CXX_FLAGS[-1],
                    *ARCH_FLAGS
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
