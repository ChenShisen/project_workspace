#include "packed_attention_kernel.cuh"
#include "cutlass_helpers.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mha_forward_dispatch(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& q_weight, const torch::Tensor& q_bias,
    const torch::Tensor& k_weight, const torch::Tensor& k_bias,
    const torch::Tensor& v_weight, const torch::Tensor& v_bias,
    const torch::Tensor& out_weight, const torch::Tensor& out_bias,
    const int num_heads)
{
    CHECK_INPUT(hidden_states);
    // ... all other checks ...
    TORCH_CHECK(hidden_states.dtype() == q_weight.dtype(), "Input and weight must have same dtype");

    const auto total_tokens = hidden_states.size(0);
    const auto hidden_size = hidden_states.size(1);
    const int head_size = hidden_size / num_heads;
    TORCH_CHECK(hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads");

    auto q_proj = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    auto k_proj = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    auto v_proj = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    
    // Step 1: QKV Projection using CUTLASS
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden_states.scalar_type(), "qkv_projection_cutlass", ([&] {
        cutlass_gemm_bias<scalar_t>(q_proj, hidden_states, q_weight, q_bias);
        cutlass_gemm_bias<scalar_t>(k_proj, hidden_states, k_weight, k_bias);
        cutlass_gemm_bias<scalar_t>(v_proj, hidden_states, v_weight, v_bias);
    }));

    auto q = q_proj.view({total_tokens, num_heads, head_size});
    auto k = k_proj.view({total_tokens, num_heads, head_size});
    auto v = v_proj.view({total_tokens, num_heads, head_size});

    // Step 2: Packed Attention using our custom kernel
    auto attn_output = torch::empty_like(q);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "packed_attention_kernel_dispatch", ([&] {
        packed_attention_cuda<scalar_t>(attn_output, q, k, v, cu_seqlens);
    }));

    auto attn_output_reshaped = attn_output.view({total_tokens, hidden_size});
    
    // Step 3: Output Projection using CUTLASS
    auto final_output = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden_states.scalar_type(), "out_projection_cutlass", ([&] {
        cutlass_gemm_bias<scalar_t>(final_output, attn_output_reshaped, out_weight, out_bias);
    }));

    return final_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mha_forward_dispatch, "Optimized MHA Forward (CUTLASS + Custom Kernel)");
}
