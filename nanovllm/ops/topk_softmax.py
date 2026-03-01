import torch
import triton
import triton.language as tl


@triton.jit
def topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    stride_l0,
    stride_l1,
    stride_w0,
    stride_w1,
    stride_i0,
    stride_i1,
    T,
    E,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask_e = cols < E

    offsets_l = pid * stride_l0 + cols * stride_l1
    logits = tl.load(logits_ptr + offsets_l, mask=mask_e, other=-float("inf")).to(
        tl.float32
    )

    k_cols = tl.arange(0, K)
    topv = tl.full((K,), -float("inf"), dtype=tl.float32)
    topi = tl.zeros((K,), dtype=tl.int16)

    for i in range(K):
        idx = tl.argmax(logits, axis=0)
        val = tl.max(logits, axis=0)
        topv = tl.where(k_cols == i, val, topv)
        topi = tl.where(k_cols == i, idx.to(tl.int16), topi)
        logits = tl.where(cols == idx, -float("inf"), logits)

    m = tl.max(topv, axis=0)
    expv = tl.exp(topv - m)
    s = expv / tl.sum(expv, axis=0)

    weights_ptrs = weights_ptr + pid * stride_w0 + k_cols * stride_w1
    indices_ptrs = indices_ptr + pid * stride_i0 + k_cols * stride_i1

    tl.store(weights_ptrs, s.to(OUT_DTYPE), mask=True)
    tl.store(indices_ptrs, topi, mask=True)


def topk_softmax(logits: torch.Tensor, topk: int):
    assert logits.is_cuda and logits.dim() == 2
    T, E = logits.shape
    assert 1 <= topk <= E

    BLOCK_SIZE = triton.next_power_of_2(E)
    grid = (T,)

    weights = torch.empty((T, topk), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((T, topk), device=logits.device, dtype=torch.int32)

    if logits.dtype == torch.float16:
        out_dtype = tl.float16
    elif logits.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif logits.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype: {logits.dtype}")

    topk_softmax_kernel[grid](
        logits,
        weights,
        indices,
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        T,
        E,
        K=topk,
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_DTYPE=out_dtype,
        num_warps=2 if BLOCK_SIZE <= 64 else (4 if BLOCK_SIZE <= 128 else 8),
    )
    return weights, indices


def reference_topk_softmax(logits, topk):
    logits_f32 = logits.to(torch.float32)
    values_f32, indices = torch.topk(logits_f32, k=topk, dim=1, sorted=True)
    maxes = values_f32.max(dim=1, keepdim=True)[0]
    expv = torch.exp(values_f32 - maxes)
    probs = expv / expv.sum(dim=1, keepdim=True)
    probs = probs.to(logits.dtype)
    return probs, indices


if __name__ == "__main__":
    torch.manual_seed(42)
    logits = torch.randn((1024, 128), dtype=torch.bfloat16, device="cuda")
    topk = 8

    weights, indices = topk_softmax(logits, topk)
    ref_weights, ref_indices = reference_topk_softmax(logits, topk)

    weights_close = torch.allclose(
        weights.float(), ref_weights.float(), atol=1e-3, rtol=1e-3
    )
    indices_equal = torch.equal(indices, ref_indices)

    print(f"Weights close: {weights_close}")
    print(f"Indices equal: {indices_equal}")

    if weights_close and indices_equal:
        print("Validation passed!")
    else:
        print("Validation failed!")
        # 打印样本和不匹配位置（用于进一步调试）
        print("Sample weights (Triton):", weights[0])
        print("Sample weights (Ref):", ref_weights[0])
        print("Sample indices (Triton):", indices[0])
        print("Sample indices (Ref):", ref_indices[0])
