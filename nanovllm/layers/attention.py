import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


def _apply_causal_mask(scores: torch.Tensor, lq: int, lk: int) -> None:
    # q are the last lq tokens in a context of length lk.
    # Allow keys up to index (i + offset) for query index i.
    offset = lk - lq
    q_idx = torch.arange(lq, device=scores.device).view(1, lq, 1)
    k_idx = torch.arange(lk, device=scores.device).view(1, 1, lk)
    mask = k_idx > (q_idx + offset)
    scores.masked_fill_(mask, float("-inf"))


def _expand_kv_heads(
    k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # k,v: [Lk, num_kv_heads, d]
    num_kv_heads = k.size(1)
    if num_kv_heads == num_heads:
        return k, v
    assert num_heads % num_kv_heads == 0
    repeat = num_heads // num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    return k, v


def _torch_varlen_attention_cudagraph_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    block_table: torch.Tensor,
) -> torch.Tensor:
    # CUDA graph path is captured with decode layout: one q token per sequence.
    bs, num_heads, head_dim = q.shape
    if bs == 0:
        return q.new_empty((0, num_heads, head_dim))
    max_num_blocks = block_table.size(1)
    block_size = k.size(1)
    max_tokens = max_num_blocks * block_size
    num_kv_heads = k.size(2)

    # Any padded entries are clamped to 0 length to avoid invalid masks.
    k_lens = cu_seqlens_k[1 : bs + 1] - cu_seqlens_k[:bs]
    k_lens = torch.clamp(k_lens, min=0, max=max_tokens)
    token_idx = torch.arange(max_tokens, device=q.device)

    out = torch.empty_like(q)
    for i in range(bs):
        block_ids = block_table[i]
        valid_blocks = block_ids >= 0
        safe_block_ids = torch.where(
            valid_blocks, block_ids, torch.zeros_like(block_ids)
        ).to(torch.long)

        k_i = k[safe_block_ids].reshape(max_tokens, num_kv_heads, head_dim)
        v_i = v[safe_block_ids].reshape(max_tokens, num_kv_heads, head_dim)
        k_i, v_i = _expand_kv_heads(k_i, v_i, num_heads)

        valid_tokens = valid_blocks[:, None].expand(max_num_blocks, block_size)
        valid_tokens = valid_tokens.reshape(max_tokens)
        len_tokens = token_idx < k_lens[i]
        attn_mask = valid_tokens & len_tokens

        scores = torch.einsum("hd,khd->hk", q[i].float(), k_i.float()) * scale
        scores.masked_fill_(~attn_mask.unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        out_i = torch.einsum("hk,khd->hd", attn, v_i.float()).to(dtype=q.dtype)

        # Keep padded rows stable (all-zero) instead of producing garbage.
        has_valid_k = ((k_lens[i] > 0) & valid_blocks.any()).to(dtype=out_i.dtype)
        out[i] = out_i * has_valid_k
    return out


def torch_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    block_table: torch.Tensor | None = None,
    causal: bool = True,
) -> torch.Tensor:
    # q: [total_q, num_heads, d]
    # k,v: [total_k, num_kv_heads, d] OR cache [num_blocks, block_size, num_kv_heads, d] when block_table is not None
    if torch.cuda.is_current_stream_capturing():
        if (
            block_table is None
            or cu_seqlens_q.numel() != q.size(0) + 1
            or cu_seqlens_k.numel() != q.size(0) + 1
        ):
            raise RuntimeError(
                "torch_varlen_attention under CUDA Graph requires decode layout "
                "(one query token per sequence) with block table."
            )
        return _torch_varlen_attention_cudagraph_decode(
            q=q,
            k=k,
            v=v,
            cu_seqlens_k=cu_seqlens_k,
            scale=scale,
            block_table=block_table,
        )

    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    bs = len(cu_q) - 1
    num_heads = q.size(1)
    out_chunks = []

    if block_table is None:
        for i in range(bs):
            q_i = q[cu_q[i] : cu_q[i + 1]]
            k_i = k[cu_k[i] : cu_k[i + 1]]
            v_i = v[cu_k[i] : cu_k[i + 1]]
            if q_i.numel() == 0:
                continue
            k_i, v_i = _expand_kv_heads(k_i, v_i, num_heads)
            q_t = q_i.transpose(0, 1)
            k_t = k_i.transpose(0, 1)
            v_t = v_i.transpose(0, 1)
            scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * scale
            if causal:
                _apply_causal_mask(scores, q_i.size(0), k_i.size(0))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_t.float()).to(dtype=q.dtype)
            out_chunks.append(out.transpose(0, 1))
    else:
        block_table_cpu = block_table.detach().cpu().tolist()
        # block_size = k.size(1)
        for i in range(bs):
            q_i = q[cu_q[i] : cu_q[i + 1]]
            lk = cu_k[i + 1] - cu_k[i]
            if q_i.numel() == 0:
                continue
            block_ids = block_table_cpu[i]
            k_blocks = []
            v_blocks = []
            for block_id in block_ids:
                if block_id < 0:
                    break
                k_blocks.append(k[block_id])
                v_blocks.append(v[block_id])
            if k_blocks:
                k_i = torch.cat(k_blocks, dim=0)[:lk]
                v_i = torch.cat(v_blocks, dim=0)[:lk]
            else:
                k_i = k.new_empty((0, k.size(2), k.size(3)))
                v_i = v.new_empty((0, v.size(2), v.size(3)))
            k_i, v_i = _expand_kv_heads(k_i, v_i, num_heads)
            q_t = q_i.transpose(0, 1)
            k_t = k_i.transpose(0, 1)
            v_t = v_i.transpose(0, 1)
            scores = torch.matmul(q_t.float(), k_t.float().transpose(-1, -2)) * scale
            if causal:
                _apply_causal_mask(scores, q_i.size(0), k_i.size(0))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_t.float()).to(dtype=q.dtype)
            out_chunks.append(out.transpose(0, 1))

    if not out_chunks:
        return q.new_empty((0, q.size(1), q.size(2)))
    return torch.cat(out_chunks, dim=0)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.block_tables is not None:  # prefix cache
            k, v = k_cache, v_cache
        o = torch_varlen_attention(
            q,
            k,
            v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            scale=self.scale,
            block_table=context.block_tables,
            causal=True,
        )
        return o
