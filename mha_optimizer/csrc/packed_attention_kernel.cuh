#pragma once

#include <torch/extension.h>

// 总调度函数接口
torch::Tensor mha_forward_dispatch(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& q_weight, const torch::Tensor& q_bias,
    const torch::Tensor& k_weight, const torch::Tensor& k_bias,
    const torch::Tensor& v_weight, const torch::Tensor& v_bias,
    const torch::Tensor& out_weight, const torch::Tensor& out_bias,
    const int num_heads
);

// 自定义Attention Kernel的模板声明
template <typename scalar_t>
void packed_attention_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens
);
