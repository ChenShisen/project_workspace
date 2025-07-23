import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- Configure CUTLASS path ---
cutlass_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cutlass'))
if not os.path.exists(cutlass_dir):
    raise RuntimeError(f"CUTLASS not found at {cutlass_dir}.")

cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools/util/include')
]

# --- Get CXX11_ABI flag from PyTorch for compatibility ---
CXX_FLAGS = ['-O3', '-std=c++17']
if torch.compiled_with_cxx11_abi():
    CXX_FLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=1')
else:
    CXX_FLAGS.append('-D_GLIBCXX_USE_CXX11_ABI=0')


# --- Select ONLY ONE architecture to minimize memory usage ---
# This is set for sm_80 (NVIDIA A100 / A10G).
# Change this if your GPU architecture is different.
ARCH_FLAGS = ['-gencode', 'arch=compute_80,code=sm_80']

setup(
    name='bert_optimizer_cpp',
    version='1.3.0', # Final stable version
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'bert_optimizer_cpp',
            [
                'csrc/packed_attention.cpp',
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
                    # Pass the ABI flag to the CUDA compiler as well
                    '-Xcompiler', CXX_FLAGS[-1],
                    *ARCH_FLAGS
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
