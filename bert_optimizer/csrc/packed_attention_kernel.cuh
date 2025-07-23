#pragma once

#include <torch/extension.h>

torch::Tensor packed_attention_dispatch(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& q_weight, const torch::Tensor& q_bias,
    const torch::Tensor& k_weight, const torch::Tensor& k_bias,
    const torch::Tensor& v_weight, const torch::Tensor& v_bias,
    const torch::Tensor& out_weight, const torch::Tensor& out_bias
);

template <typename scalar_t>
void packed_attention_forward_cuda_template(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens
);
