#pragma once

#include <torch/extension.h>

// 总调度函数，它将编排CUTLASS GEMM和我们的自定义Attention Kernel
torch::Tensor packed_attention_dispatch(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& q_weight, const torch::Tensor& q_bias,
    const torch::Tensor& k_weight, const torch::Tensor& k_bias,
    const torch::Tensor& v_weight, const torch::Tensor& v_bias,
    const torch::Tensor& out_weight, const torch::Tensor& out_bias
);

// 自定义Attention Kernel的模板声明，定义在.cu文件中
template <typename scalar_t>
void packed_attention_forward_cuda_template(
    torch::Tensor& out,         // [total_tokens, num_heads, head_size]
    const torch::Tensor& q,     // [total_tokens, num_heads, head_size]
    const torch::Tensor& k,     // [total_tokens, num_heads, head_size]
    const torch::Tensor& v,     // [total_tokens, num_heads, head_size]
    const torch::Tensor& cu_seqlens // [batch_size + 1]
);
