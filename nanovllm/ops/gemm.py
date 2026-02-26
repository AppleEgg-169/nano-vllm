import torch
import triton
import triton.language as tl
import triton.testing as tt
import torch.cuda.nvtx as nvtx

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    stride_c0,
    stride_c1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_group = group_id * GROUP_SIZE_M
    rela_pos = pid % num_pid_group
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_group)

    pid_m = first_pid_group + rela_pos % group_size_m
    pid_n = rela_pos // group_size_m

    offsets_a = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_b = BLOCK_SIZE_N * pid_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offsets_a[:, None] * stride_a0 + offsets_k[None, :] * stride_a1)
    b_ptrs = b_ptr + (offsets_k[:, None] * stride_b0 + offsets_b[None, :] * stride_b1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_a = (offsets_a[:, None] < M) & (offsets_k[None, :] + k < K)
        mask_b = (offsets_k[:, None] + k < K) & (offsets_b[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_a1
        b_ptrs += BLOCK_SIZE_K * stride_b0

    offsets_c0 = offsets_a
    offsets_c1 = offsets_b

    c_ptrs = c_ptr + (offsets_c0[:, None] * stride_c0 + offsets_c1[None, :] * stride_c1)
    mask_c = (offsets_c0[:, None] < M) & (offsets_c1[None, :] < N)
    res = acc
    tl.store(c_ptrs, res.to(tl.float16), mask=mask_c)


def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
):
    M, K = a.shape
    _, N = b.shape

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )


def benchmark_gemm(M, N, K):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c_triton = torch.empty((M, N), device="cuda", dtype=torch.float16)
    c_cublas = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Triton correctness check
    solve(a, b, c_triton)
    torch.cuda.synchronize()
    torch_output = torch.matmul(a, b)
    is_correct_triton = torch.allclose(c_triton, torch_output, atol=1e-2, rtol=1e-2)
    print(f"Triton Correctness: {'PASSED' if is_correct_triton else 'FAILED'}")

    # cuBLAS correctness (should always pass)
    c_cublas = torch.matmul(a, b)
    is_correct_cublas = torch.allclose(c_cublas, torch_output, atol=1e-2, rtol=1e-2)
    print(f"cuBLAS Correctness: {'PASSED' if is_correct_cublas else 'FAILED'}")

    # Benchmark Triton
    print("\nBenchmarking Triton...")
    with nvtx.range(f"bench_triton M:{M} N:{N} K:{K}"):
        # Warmup
        for _ in range(20):
            solve(a, b, c_triton)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            solve(a, b, c_triton)
        end.record()
        end.synchronize()
        avg_time_triton = start.elapsed_time(end) / 100  # ms

    flops = 2 * M * N * K
    tflops_triton = flops / (avg_time_triton * 1e9) if avg_time_triton > 0 else 0
    bytes_transferred = (M * K + K * N + M * N) * 2  # fp16: 2 bytes per element
    bandwidth_triton = bytes_transferred / (avg_time_triton * 1e6) if avg_time_triton > 0 else 0

    print(f"Triton - Avg time: {avg_time_triton:.2f} ms")
    print(f"Triton - TFLOPS: {tflops_triton:.2f}")
    print(f"Triton - Bandwidth: {bandwidth_triton:.2f} GB/s")

    # Benchmark cuBLAS
    print("\nBenchmarking cuBLAS...")
    with nvtx.range(f"bench_cublas M:{M} N:{N} K:{K}"):
        # Warmup
        for _ in range(20):
            c_cublas = torch.matmul(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            c_cublas = torch.matmul(a, b)
        end.record()
        end.synchronize()
        avg_time_cublas = start.elapsed_time(end) / 100  # ms

    tflops_cublas = flops / (avg_time_cublas * 1e9) if avg_time_cublas > 0 else 0
    bandwidth_cublas = bytes_transferred / (avg_time_cublas * 1e6) if avg_time_cublas > 0 else 0

    print(f"cuBLAS - Avg time: {avg_time_cublas:.2f} ms")
    print(f"cuBLAS - TFLOPS: {tflops_cublas:.2f}")
    print(f"cuBLAS - Bandwidth: {bandwidth_cublas:.2f} GB/s")

    # Comparison
    print("\nComparison:")
    print(f"Triton vs cuBLAS - Time Ratio: {avg_time_triton / avg_time_cublas:.2f}x")
    print(f"Triton vs cuBLAS - TFLOPS Ratio: {tflops_triton / tflops_cublas * 100:.1f}%")
    print(f"Triton vs cuBLAS - Bandwidth Ratio: {bandwidth_triton / bandwidth_cublas * 100:.1f}%")

# Run benchmarks
print("Benchmark for (128, 128, 128):")
benchmark_gemm(128, 128, 128)

print("\nBenchmark for (1024, 1024, 1024):")
benchmark_gemm(1024, 1024, 1024)

print("\nBenchmark for (4096, 4096, 4096):")
benchmark_gemm(4096, 4096, 4096)