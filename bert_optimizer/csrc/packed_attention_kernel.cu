#include "packed_attention_kernel.cuh"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <limits>

// --- Helper to get infinity for different types ---
template <typename T>
__device__ __forceinline__ T Infinity();

template <>
__device__ __forceinline__ float Infinity<float>() {
    return std::numeric_limits<float>::infinity();
}
template <>
__device__ __forceinline__ half Infinity<half>() {
    // a half representation of +INF
    return __ushort_as_half(0x7C00);
}

// --- Packed Attention Kernel Implementation ---
// Parallelization Strategy:
// Grid: (num_heads, batch_size) -> One CUDA Block per (head, sequence) pair.
// Block: (e.g., 128 threads) -> Threads within a block are parallelized over tokens in the sequence.
template <typename scalar_t, int HeadSize, int BlockSizeT>
__global__ void packed_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int* __restrict__ cu_seqlens,
    const int num_heads)
{
    // --- 1. Identify work for this block ---
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;

    const int seq_start_token = cu_seqlens[seq_idx];
    const int seq_end_token = cu_seqlens[seq_idx + 1];
    const int context_len = seq_end_token - seq_start_token;

    // If sequence is empty, do nothing.
    if (context_len == 0) return;

    // --- 2. Parallelize over tokens in the sequence ---
    // Each thread processes one or more tokens in this sequence using a grid-stride loop.
    for(int token_offset = threadIdx.x; token_offset < context_len; token_offset += blockDim.x) {
        const int global_token_idx = seq_start_token + token_offset;

        // --- 3. Online Softmax Logic for each token ---
        float m = -Infinity<float>();
        float l = 0.0f;
        float acc[HeadSize];
        for (int i = 0; i < HeadSize; ++i) acc[i] = 0.0f;

        // Pointer to this token's Q vector for the current head
        const scalar_t* my_q = &q[(global_token_idx * num_heads + head_idx) * HeadSize];
        const float scale = 1.0f / rsqrtf(static_cast<float>(HeadSize));

        // This token must attend to all previous tokens in its sequence (causal attention)
        const int causal_len = token_offset + 1;

        // --- 4. Tiling over the key/value history ---
        for (int j_start = 0; j_start < causal_len; j_start += BlockSizeT) {
            int j_end = min(j_start + BlockSizeT, causal_len);

            // Allocate shared memory for a tile of K and V
            extern __shared__ scalar_t smem[];
            scalar_t* s_k = smem; // Shape: [BlockSizeT, HeadSize]

            // --- Load a tile of K into shared memory ---
            __syncthreads(); // Make sure previous V tile usage is done
            for(int j = 0; j < j_end - j_start; ++j) {
                const int history_token_idx = seq_start_token + j_start + j;
                for(int i = 0; i < HeadSize; ++i) {
                   s_k[j * HeadSize + i] = k[(history_token_idx * num_heads + head_idx) * HeadSize + i];
                }
            }
            __syncthreads();

            // --- Compute scores for this tile ---
            float m_tile = -Infinity<float>();
            float scores[BlockSizeT];
            for (int j = 0; j < j_end - j_start; ++j) {
                float score = 0.0f;
                for (int i = 0; i < HeadSize; ++i) {
                    score += static_cast<float>(my_q[i]) * static_cast<float>(s_k[j * HeadSize + i]);
                }
                scores[j] = score * scale;
                m_tile = fmaxf(m_tile, scores[j]);
            }

            // --- Online Softmax Update (numerically stable) ---
            float m_old = m;
            m = fmaxf(m, m_tile);

            float l_prime = 0.0f;
            for (int j = 0; j < j_end - j_start; ++j) {
                float p = expf(scores[j] - m);
                scores[j] = p; // Store exp(score - new_max) to reuse
                l_prime += p;
            }

            float l_new = l * expf(m_old - m) + l_prime;

            for (int i = 0; i < HeadSize; ++i) {
                acc[i] *= expf(m_old - m);
            }
            l = l_new;

            // --- Load V tile and accumulate results ---
            __syncthreads(); // Wait for all threads to finish with K tile
            scalar_t* s_v = smem; // Reuse shared memory for V tile
            for(int j = 0; j < j_end - j_start; ++j) {
                const int history_token_idx = seq_start_token + j_start + j;
                for(int i = 0; i < HeadSize; ++i) {
                   s_v[j * HeadSize + i] = v[(history_token_idx * num_heads + head_idx) * HeadSize + i];
                }
            }
            __syncthreads();

            for (int j = 0; j < j_end - j_start; ++j) {
                float p = scores[j];
                for (int i = 0; i < HeadSize; ++i) {
                    acc[i] += p * static_cast<float>(s_v[j * HeadSize + i]);
                }
            }
        }

        // --- 5. Write final normalized output for this token ---
        scalar_t* my_out = &out[(global_token_idx * num_heads + head_idx) * HeadSize];
        float inv_l = 1.0f / l;
        for (int i = 0; i < HeadSize; ++i) {
            my_out[i] = static_cast<scalar_t>(acc[i] * inv_l);
        }
    }
}


// --- C++ Wrapper that launches the kernel ---
// This is where we instantiate our kernel templates for specific data types and sizes.
template <typename scalar_t>
void packed_attention_forward_cuda_template(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens) {

    const int batch_size = cu_seqlens.size(0) - 1;
    const int num_heads = q.size(1);
    const int head_size = q.size(2);

    // This kernel is compiled for a specific HeadSize.
    // A production library would use a dispatcher to select from multiple compiled versions.
    TORCH_CHECK(head_size == 64, "This kernel is compiled for head_size=64 only.");
    constexpr int HeadSize = 64;
    // Tiling size for K/V sequence length. Can be tuned.
    constexpr int BlockSizeT = 64;

    dim3 grid(num_heads, batch_size);
    dim3 block(128); // Number of threads per block

    // Shared memory size: enough to hold one tile of K or V.
    const size_t shared_mem_size = BlockSizeT * HeadSize * sizeof(scalar_t);

    // Launch the kernel
    packed_attention_kernel<scalar_t, HeadSize, BlockSizeT><<<grid, block, shared_mem_size>>>(
        out.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        cu_seqlens.data_ptr<int>(),
        num_heads
    );

    // Check for any errors during kernel launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Packed Attention Kernel launch failed");
}

// --- Explicitly instantiate templates for supported types (float and half) ---
// This tells the compiler to generate code for these specific types.
template void packed_attention_forward_cuda_template<float>(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
template void packed_attention_forward_cuda_template<at::Half>(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
