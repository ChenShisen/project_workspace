import torch
import torch.nn as nn
import time
import random
import warnings

try:
    import bert_optimizer_cpp
except ImportError:
    warnings.warn(
        "Could not import 'bert_optimizer_cpp'. "
        "Please make sure the extension is built correctly by running 'pip install .' in the project root."
    )
    exit()

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
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.out_proj(attn_output)

class OptimizedPackedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, hidden_states_packed, cu_seqlens):
        return bert_optimizer_cpp.forward(
            hidden_states_packed, cu_seqlens,
            self.q_proj.weight, self.q_proj.bias,
            self.k_proj.weight, self.k_proj.bias,
            self.v_proj.weight, self.v_proj.bias,
            self.out_proj.weight, self.out_proj.bias,
        )

def run_benchmark_for_dtype(dtype_str):
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping benchmark.")
        return
    device = "cuda"
    dtype = torch.float32 if dtype_str == "fp32" else torch.float16
    
    print("\n" + "="*60)
    print(f"        RUNNING BENCHMARK FOR {dtype_str.upper()}")
    print("="*60)
    
    class Config:
        hidden_size = 768
        num_attention_heads = 12
        head_size = 64
    config = Config()
    assert config.hidden_size == config.num_attention_heads * config.head_size

    baseline_model = StandardBertAttention(config).to(device).eval()
    optimized_model = OptimizedPackedAttention(config).to(device).eval()
    optimized_model.load_state_dict(baseline_model.state_dict())
    
    if dtype == torch.float16:
        baseline_model.half()
        optimized_model.half()

    batch_size = 32
    seq_lens = [random.randint(128, 512) for _ in range(batch_size)]
    max_len = max(seq_lens)
    
    padded_input = torch.randn(batch_size, max_len, config.hidden_size, device=device, dtype=dtype)
    unpacked_inputs = [padded_input[i, :l, :].clone() for i, l in enumerate(seq_lens)]
    packed_input = torch.cat(unpacked_inputs, dim=0)
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), device=device, dtype=torch.int32)
    
    print("\n--- Running Baseline (Padded PyTorch SDPA)...")
    with torch.no_grad():
        for _ in range(20): _ = baseline_model(padded_input)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100): baseline_output = baseline_model(padded_input)
        torch.cuda.synchronize()
        end_time = time.time()
    baseline_latency = (end_time - start_time) * 1000 / 100
    print(f"Baseline Latency: {baseline_latency:.4f} ms")

    print("\n--- Running Optimized (Packed, CUTLASS + Custom Kernel)...")
    with torch.no_grad():
        for _ in range(20): _ = optimized_model(packed_input, cu_seqlens)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100): optimized_output_packed = optimized_model(packed_input, cu_seqlens)
        torch.cuda.synchronize()
        end_time = time.time()
    optimized_latency = (end_time - start_time) * 1000 / 100
    print(f"Optimized Kernel Latency: {optimized_latency:.4f} ms")

    print("\n--- Verification and Results ---")
    baseline_output_unpacked = torch.cat([baseline_output[i, :l, :] for i, l in enumerate(seq_lens)], dim=0)
    
    rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-4, 1e-4)
    is_close = torch.allclose(optimized_output_packed, baseline_output_unpacked, rtol=rtol, atol=atol)
    print(f"Correctness Check: {'PASSED' if is_close else 'FAILED'}")
    if not is_close:
        abs_err = (optimized_output_packed - baseline_output_unpacked).abs()
        print(f"  Max Absolute Error: {abs_err.max().item():.6f}")

    speedup = baseline_latency / optimized_latency
    print("\n" + "-"*30)
    print(f"Overall Speedup for {dtype_str.upper()}: {speedup:.2f}x")
    print("-"*30)

if __name__ == "__main__":
    run_benchmark_for_dtype("fp32")
    run_benchmark_for_dtype("fp16")
