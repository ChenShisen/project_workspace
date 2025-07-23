import torch
import torch.nn as nn
import time
import random
import warnings

# --- 导入我们的自定义C++扩展 ---
# 如果编译成功，这个导入就会起作用
try:
    import bert_optimizer_cpp
except ImportError:
    warnings.warn(
        "Could not import 'bert_optimizer_cpp'. "
        "Please make sure the extension is built correctly by running 'pip install .' "
        "in the project root directory."
    )
    exit()

# --- 1. 定义一个标准的PyTorch BertAttention作为基准 ---
class StandardBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # 使用PyTorch内置的高度优化的scaled_dot_product_attention
        # is_causal=True 会自动生成因果掩码，这正是我们想要的
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

# --- 2. 定义我们优化的、使用C++扩展的模块 ---
class OptimizedPackedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 我们需要线性层的权重和偏置来传递给我们的C++函数
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states_packed, cu_seqlens):
        # 我们的C++函数将处理从投影到Attention再到输出投影的所有事情
        # 注意: PyTorch的nn.Linear权重形状是(out_features, in_features)，
        # 而我们的CUTLASS helper期望的权重形状也是(out_features, in_features)。
        return bert_optimizer_cpp.forward(
            hidden_states_packed,
            cu_seqlens,
            self.q_proj.weight, self.q_proj.bias,
            self.k_proj.weight, self.k_proj.bias,
            self.v_proj.weight, self.v_proj.bias,
            self.out_proj.weight, self.out_proj.bias,
        )

# --- 3. 基准测试主函数 ---
def run_benchmark_for_dtype(dtype_str):
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping benchmark.")
        return

    device = "cuda"
    dtype = torch.float32 if dtype_str == "fp32" else torch.float16
    
    print("\n" + "="*60)
    print(f"        RUNNING BENCHMARK FOR {dtype_str.upper()}")
    print("="*60)

    # --- 模型和配置 ---
    class Config:
        hidden_size = 768
        num_attention_heads = 12
        # 我们的Kernel是为head_size=64硬编码编译的
        head_size = 64
    config = Config()
    assert config.hidden_size == config.num_attention_heads * config.head_size

    baseline_model = StandardBertAttention(config).to(device).eval()
    optimized_model = OptimizedPackedAttention(config).to(device).eval()
    
    # 确保两个模型使用完全相同的权重，以进行公平的正确性比较
    optimized_model.load_state_dict(baseline_model.state_dict())
    
    if dtype == torch.float16:
        baseline_model.half()
        optimized_model.half()

    # --- 准备输入数据 ---
    batch_size = 32
    # 生成一组随机变化的序列长度，以模拟真实场景
    seq_lens = [random.randint(128, 512) for _ in range(batch_size)]
    max_len = max(seq_lens)
    
    # 为基准模型准备数据 (填充)
    padded_input = torch.randn(batch_size, max_len, config.hidden_size, device=device, dtype=dtype)

    # 为优化模型准备数据 (压实)
    unpacked_inputs = [padded_input[i, :l, :].clone() for i, l in enumerate(seq_lens)]
    packed_input = torch.cat(unpacked_inputs, dim=0)
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), device=device, dtype=torch.int32)
    
    # --- 基准测试: Baseline ---
    print("\n--- Running Baseline (Padded PyTorch SDPA)...")
    with torch.no_grad():
        # 预热
        for _ in range(20):
            _ = baseline_model(padded_input)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            baseline_output = baseline_model(padded_input)
        torch.cuda.synchronize()
        end_time = time.time()
    
    baseline_latency = (end_time - start_time) / 100 * 1000 # ms per call
    print(f"Baseline Latency: {baseline_latency:.4f} ms")

    # --- 基准测试: Optimized ---
    print("\n--- Running Optimized (Packed, CUTLASS + Custom Kernel)...")
    with torch.no_grad():
        # 预热
        for _ in range(20):
            _ = optimized_model(packed_input, cu_seqlens)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            optimized_output_packed = optimized_model(packed_input, cu_seqlens)
        torch.cuda.synchronize()
        end_time = time.time()

    optimized_latency = (end_time - start_time) / 100 * 1000 # ms per call
    print(f"Optimized Kernel Latency: {optimized_latency:.4f} ms")

    # --- 结果验证 ---
    print("\n--- Verification and Results ---")
    
    # 将基准模型的填充输出也进行压实，以便比较
    baseline_output_unpacked = torch.cat([baseline_output[i, :l, :] for i, l in enumerate(seq_lens)], dim=0)
    
    # 正确性检查
    try:
        # FP16的容忍度需要放宽一些
        rtol = 1e-2 if dtype == torch.float16 else 1e-3
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        is_close = torch.allclose(optimized_output_packed, baseline_output_unpacked, rtol=rtol, atol=atol)
        
        print(f"Correctness Check: {'PASSED' if is_close else 'FAILED'}")
        if not is_close:
            abs_err = (optimized_output_packed - baseline_output_unpacked).abs()
            rel_err = abs_err / baseline_output_unpacked.abs()
            print(f"  Max Absolute Error: {abs_err.max().item():.6f}")
            print(f"  Max Relative Error: {rel_err.max().item():.6f}")

    except Exception as e:
         print(f"Correctness Check: FAILED with error: {e}")
    
    speedup = baseline_latency / optimized_latency
    print("\n" + "-"*30)
    print(f"Overall Speedup for {dtype_str.upper()}: {speedup:.2f}x")
    print("-"*30)


if __name__ == "__main__":
    run_benchmark_for_dtype("fp32")
    run_benchmark_for_dtype("fp16")
