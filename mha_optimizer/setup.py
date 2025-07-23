import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- 1. 配置CUTLASS路径 ---
cutlass_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cutlass'))
if not os.path.exists(cutlass_dir):
    raise RuntimeError(f"CUTLASS not found at {cutlass_dir}.")

cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools/util/include')
]

# --- 2. 获取ABI兼容性标志 ---
CXX_FLAGS = ['-O3', '-std=c++17']
if torch.compiled_with_cxx11_abi():
    CXX_FLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=1')
else:
    CXX_FLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=0')

# --- 3. 只为你的GPU架构编译 (以sm_80为例) ---
ARCH_FLAGS = ['-gencode', 'arch=compute_80,code=sm_80']

setup(
    name='mha_optimizer_cpp',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'mha_optimizer_cpp',
            [
                'csrc/mha_optimizer.cpp',
                'csrc/packed_attention_kernel.cu',
            ],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={
                'cxx': CXX_FLAGS,
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
                    '-Xcompiler', CXX_FLAGS[-1],
                    *ARCH_FLAGS
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
