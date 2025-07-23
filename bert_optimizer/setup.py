import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- 配置 CUTLASS 路径 ---
# 这个脚本期望 cutlass 仓库与 bert_optimizer 项目在同一个父目录下
cutlass_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cutlass'))

# 检查CUTLASS路径是否存在
if not os.path.exists(cutlass_dir):
    raise RuntimeError(
        f"CUTLASS not found at {cutlass_dir}. "
        "Please clone it from https://github.com/NVIDIA/cutlass.git into the parent directory."
    )

# 定义CUTLASS的头文件包含路径
cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools/util/include')
]

print("--- Found CUTLASS at: ", cutlass_dir)
print("--- Using CUTLASS include directories: ", cutlass_include_dirs)

setup(
    name='bert_optimizer_cpp',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            # 扩展模块的名称，Python中将通过这个名字导入
            'bert_optimizer_cpp',
            # 源文件列表
            [
                'csrc/packed_attention.cpp',
                'csrc/packed_attention_kernel.cu',
            ],
            # 添加CUTLASS的头文件路径
            include_dirs=cutlass_include_dirs,
            # 额外的编译参数
            extra_compile_args={
                'cxx': [
                    '-O3',                # 优化等级
                    '-std=c++17',         # C++标准
                    '-Wall',              # 开启所有警告
                    '-Wextra',            # 开启额外的警告
                ],
                'nvcc': [
                    '-O3',                # 优化等级
                    '--use_fast_math',    # 使用快速数学函数
                    '-std=c++17',         # C++标准
                    # 开启Tensor Core，这对于FP16/BF16性能至关重要
                    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
                    # 为不同的GPU架构生成代码，以保证兼容性和性能
                    # 你可以根据你的GPU删减或增加
                    '-gencode', 'arch=compute_70,code=sm_70', # Volta (V100)
                    '-gencode', 'arch=compute_75,code=sm_75', # Turing (RTX 20x0)
                    '-gencode', 'arch=compute_80,code=sm_80', # Ampere (A100, RTX 30x0)
                    '-gencode', 'arch=compute_86,code=sm_86', # Ampere (RTX 3090)
                ]
            }
        )
    ],
    # 指定使用BuildExtension来处理C++扩展
    cmdclass={
        'build_ext': BuildExtension
    }
)
