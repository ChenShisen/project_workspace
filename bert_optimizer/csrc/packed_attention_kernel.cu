#include "packed_attention_kernel.cuh"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <limits>

template <typename T>
__device__ __forceinline__ T CudaNegativeInfinity();
template <> __device__ __forceinline__ float CudaNegativeInfinity<float>() { return -std::numeric_limits<float>::infinity(); }
template <> __device__ __forceinline__ half CudaNegativeInfinity<half>() { return __ushort_as_half(0xFC00); }

template <typename scalar_t, int HeadSize, int TileSize>
__global__ void packed_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int* __restrict__ cu_seqlens,
    const int num_heads)
{
    extern __shared__ char smem_char[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_char);

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;

    const int seq_start_token = cu_seqlens[seq_idx];
    const int seq_end_token = cu_seqlens[seq_idx + 1];
    const int context_len = seq_end_token - seq_start_token;

    if (context_len == 0) return;

    for(int token_offset = threadIdx.x; token_offset < context_len; token_offset += blockDim.x) {
        const int global_token_idx = seq_start_token + token_offset;
        
        float m = CudaNegativeInfinity<float>();
        float l = 0.0f;
        float acc[HeadSize];
        for (int i = 0; i < HeadSize; ++i) acc[i] = 0.0f;

        const scalar_t* my_q = &q[(global_token_idx * num_heads + head_idx) * HeadSize];
        const float scale = rsqrtf(static_cast<float>(HeadSize));

        const int causal_len = token_offset + 1;
        for (int tile_start = 0; tile_start < causal_len; tile_start += TileSize) {
            int tile_end = min(tile_start + TileSize, causal_len);
            
            scalar_t* s_k = smem;
            __syncthreads();
            for(int j = 0; j < tile_end - tile_start; ++j) {
                const int history_token_idx = seq_start_token + tile_start + j;
                for(int i = 0; i < HeadSize; ++i) {
                   s_k[j * HeadSize + i] = k[(history_token_idx * num_heads + head_idx) * HeadSize + i];
                }
            }
            __syncthreads();

            float m_tile = CudaNegativeInfinity<float>();
            float scores[TileSize];
            for (int j = 0; j < tile_end - tile_start; ++j) {
                float score = 0.0f;
                for (int i = 0; i < HeadSize; ++i) {
                    score += static_cast<float>(my_q[i]) * static_cast<float>(s_k[j * HeadSize + i]);
                }
                scores[j] = score * scale;
                m_tile = fmaxf(m_tile, scores[j]);
            }

            float m_old = m;
            m = fmaxf(m, m_tile);
            
            float l_prime = 0.0f;
            for (int j = 0; j < tile_end - tile_start; ++j) {
                float p = expf(scores[j] - m);
                scores[j] = p;
                l_prime += p;
            }

            float rescale_factor = expf(m_old - m);
            l = l * rescale_factor + l_prime;

            for (int i = 0; i < HeadSize; ++i) {
                acc[i] *= rescale_factor;
            }
            
            scalar_t* s_v = smem;
            __syncthreads();
            for(int j = 0; j < tile_end - tile_start; ++j) {
                const int history_token_idx = seq_start_token + tile_start + j;
                for(int i = 0; i < HeadSize; ++i) {
                   s_v[j * HeadSize + i] = v[(history_token_idx * num_heads + head_idx) * HeadSize + i];
                }
            }
            __syncthreads();

            for (int j = 0; j < tile_end - tile_start; ++j) {
                float p = scores[j];
                for (int i = 0; i < HeadSize; ++i) {
                    acc[i] += p * static_cast<float>(s_v[j * HeadSize + i]);
                }
            }
        }

        scalar_t* my_out = &out[(global_token_idx * num_heads + head_idx) * HeadSize];
        float inv_l = 1.0f / l;
        for (int i = 0; i < HeadSize; ++i) {
            my_out[i] = static_cast<scalar_t>(acc[i] * inv_l);
        }
    }
}

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
    
    TORCH_CHECK(head_size == 64, "This kernel is compiled for head_size=64 only.");
    constexpr int HeadSize = 64;
    constexpr int TileSize = 64;

    dim3 grid(num_heads, batch_size);
    dim3 block(128);
    
    const size_t shared_mem_size = TileSize * HeadSize * sizeof(scalar_t);

    packed_attention_kernel<scalar_t, HeadSize, TileSize><<<grid, block, shared_mem_size>>>(
        out.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        cu_seqlens.data_ptr<int>(),
        num_heads
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Packed Attention Kernel launch failed");
}

template void packed_attention_forward_cuda_template<float>(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
template void packed_attention_forward_cuda_template<at::Half>(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
