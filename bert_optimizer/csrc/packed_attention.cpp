#include "packed_attention_kernel.cuh"
#include "cutlass_helpers.cuh"

// --- 宏定义，用于输入检查 ---
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// --- 总调度函数实现 ---
torch::Tensor packed_attention_dispatch(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& q_weight, const torch::Tensor& q_bias,
    const torch::Tensor& k_weight, const torch::Tensor& k_bias,
    const torch::Tensor& v_weight, const torch::Tensor& v_bias,
    const torch::Tensor& out_weight, const torch::Tensor& out_bias)
{
    // --- 1. 输入验证 ---
    CHECK_INPUT(hidden_states);
    CHECK_INPUT(cu_seqlens);
    CHECK_INPUT(q_weight); CHECK_INPUT(q_bias);
    CHECK_INPUT(k_weight); CHECK_INPUT(k_bias);
    CHECK_INPUT(v_weight); CHECK_INPUT(v_bias);
    CHECK_INPUT(out_weight); CHECK_INPUT(out_bias);
    TORCH_CHECK(hidden_states.dtype() == q_weight.dtype(), "Input and weight must have same dtype");

    // --- 2. 获取维度信息 ---
    const auto total_tokens = hidden_states.size(0);
    const auto hidden_size = hidden_states.size(1);
    const auto num_heads = q_weight.size(0) / (hidden_size / 12); // Infer num_heads, assuming BERT-Base
    const auto head_size = hidden_size / num_heads;
    
    // 检查head_size是否有效
    TORCH_CHECK(hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads");

    // --- 3. QKV投影 ---
    // 为Q, K, V的投影结果分配内存
    auto q_proj = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    auto k_proj = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    auto v_proj = torch::empty({total_tokens, hidden_size}, hidden_states.options());

    // 使用AT_DISPATCH宏根据数据类型调用正确的CUTLASS模板实例
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden_states.scalar_type(), "qkv_projection_cutlass", ([&] {
        cutlass_gemm_bias<scalar_t>(q_proj, hidden_states, q_weight, q_bias);
        cutlass_gemm_bias<scalar_t>(k_proj, hidden_states, k_weight, k_bias);
        cutlass_gemm_bias<scalar_t>(v_proj, hidden_states, v_weight, v_bias);
    }));

    // --- 4. Reshape Q,K,V 为Attention准备 ---
    // [total_tokens, hidden_size] -> [total_tokens, num_heads, head_size]
    auto q = q_proj.view({total_tokens, num_heads, head_size});
    auto k = k_proj.view({total_tokens, num_heads, head_size});
    auto v = v_proj.view({total_tokens, num_heads, head_size});

    // --- 5. 调用自定义的Packed Attention Kernel ---
    // 为Attention的输出分配内存
    auto attn_output = torch::empty_like(q);

    // 使用AT_DISPATCH宏调用正确的Attention Kernel模板实例
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "packed_attention_kernel_dispatch", ([&] {
        packed_attention_forward_cuda_template<scalar_t>(attn_output, q, k, v, cu_seqlens);
    }));

    // --- 6. Reshape Attention输出，为最终投影做准备 ---
    // [total_tokens, num_heads, head_size] -> [total_tokens, hidden_size]
    auto attn_output_reshaped = attn_output.view({total_tokens, hidden_size});

    // --- 7. 调用CUTLASS进行最终的输出投影 ---
    auto final_output = torch::empty({total_tokens, hidden_size}, hidden_states.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hidden_states.scalar_type(), "out_projection_cutlass", ([&] {
        cutlass_gemm_bias<scalar_t>(final_output, attn_output_reshaped, out_weight, out_bias);
    }));

    return final_output;
}

// --- 8. 将C++函数绑定到Python模块 ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &packed_attention_dispatch, "Industry-standard Packed Attention Forward (CUTLASS + Custom Kernel)");
}
