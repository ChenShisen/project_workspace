#pragma once

#include <torch/extension.h>

// 检查依赖的头文件是否存在
#if __has_include("cutlass/gemm/device/gemm.h")
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#else
#error "CUTLASS headers not found. Make sure the include paths are set correctly in setup.py"
#endif

// 一个辅助函数，用于运行一个融合了偏置加法的CUTLASS GEMM操作。
// 这个操作实现了: C = A * B + bias
// A: 输入激活 [M, K]
// B: 权重矩阵 [N, K] (在PyTorch中是[out_features, in_features])
// C: 输出矩阵 [M, N]
// bias: 偏置向量 [N]
template<typename scalar_t>
void cutlass_gemm_bias(
    torch::Tensor& C,             // 输出张量, 形状 [M, N]
    const torch::Tensor& A,       // 输入张量, 形状 [M, K]
    const torch::Tensor& B,       // 权重张量, 形状 [N, K]
    const torch::Tensor& bias     // 偏置张量, 形状 [N]
) {
    // --- 1. 定义CUTLASS所需的类型和布局 ---
    // 计算时使用的累加类型，通常用FP32以保证精度
    using ElementCompute = float;
    // 输入/输出的数据类型
    using ElementOutput = scalar_t;
    // 内存布局，PyTorch张量默认是行主序
    using LayoutA = cutlass::layout::RowMajor;
    // B是权重，PyTorch的nn.Linear权重是(out_features, in_features)，
    // 为了匹配A*B^T的数学形式，我们将B视为列主序。
    // A[M,K] * B[K,N] -> C[M,N]  (数学形式)
    // A_row[M,K] * B_col[N,K] -> C_row[M,N] (CUTLASS布局)
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // --- 2. 定义GEMM操作 ---
    // 这个模板定义了一个完整的GEMM操作，包括线程块大小、warp大小、指令形状等。
    // 我们使用默认配置，CUTLASS会为我们选择一个合理的配置。
    // 重要的是，它的Epilogue（尾声）部分支持线性缩放和偏置加法。
    using Gemm = cutlass::gemm::device::Gemm<
        scalar_t,      LayoutA, // A的类型和布局
        scalar_t,      LayoutB, // B的类型和布局
        ElementOutput, LayoutC, // C的类型和布局
        ElementCompute         // 累加类型
    >;

    // 实例化GEMM操作
    Gemm gemm_op;

    // --- 3. 获取问题的维度和数据指针 ---
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0); // B.size(0) is out_features

    // 检查维度是否匹配
    TORCH_CHECK(B.size(1) == K, "Matrix B dimensions are incorrect");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Matrix C dimensions are incorrect");
    TORCH_CHECK(bias.size(0) == N, "Bias dimensions are incorrect");

    // 获取指向GPU内存的原始指针
    scalar_t* ptr_A = A.data_ptr<scalar_t>();
    scalar_t* ptr_B = B.data_ptr<scalar_t>();
    scalar_t* ptr_C = C.data_ptr<scalar_t>();
    scalar_t* ptr_bias = bias.data_ptr<scalar_t>();

    // --- 4. 配置GEMM参数 ---
    // CUTLASS的GEMM参数结构体
    // D = alpha * A * B + beta * C
    // 在我们的场景中: final_C = 1.0 * A * B + 1.0 * bias_broadcasted
    typename Gemm::Arguments args(
        {M, N, K},                          // GEMM问题维度 (M, N, K)
        {ptr_A, LayoutA::packed({M, K}).stride(0)}, // A张量的指针和ldm (leading dimension)
        {ptr_B, LayoutB::packed({N, K}).stride(0)}, // B张量的指针和ldm
        {ptr_bias, 0},                      // C张量(我们的偏置)的指针和ldm。ldm=0表示广播
        {ptr_C, LayoutC::packed({M, N}).stride(0)}, // D张量(我们的输出)的指针和ldm
        {1.f, 1.f}                          // alpha = 1.0, beta = 1.0
    );

    // --- 5. 检查可用性并执行GEMM ---
    // 检查我们配置的GEMM是否可以在当前设备上运行
    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM cannot be implemented for the given problem size");

    // 执行GEMM操作
    status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM launch failed");
}
