#pragma once

#include <torch/extension.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

template<typename scalar_t>
void cutlass_gemm_bias(
    torch::Tensor& D,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias
) {
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        scalar_t, 1, ElementCompute, ElementCompute>;
    
    using Gemm = cutlass::gemm::device::GemmUniversal<
        scalar_t, LayoutA,
        scalar_t, LayoutB,
        scalar_t, LayoutD,
        ElementCompute,
        EpilogueOp
    >;

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    TORCH_CHECK(B.size(1) == K, "Matrix B dimension mismatch for GEMM");

    Gemm gemm_op;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {1.f, 1.f},
        A.data_ptr(), B.data_ptr(), bias.data_ptr(), D.data_ptr(),
        0, 0, 0, 0,
        K, K, 0, N
    );

    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GemmUniversal cannot implement this problem size.");
    status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GemmUniversal launch failed");
}
