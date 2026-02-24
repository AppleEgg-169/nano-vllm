import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _moe_scatter_add_kernel(
    src_ptr,
    token_idx_ptr,
    out_ptr,
    stride_src0,
    stride_src1,
    stride_out0,
    stride_out1,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_dim

    token_idx = tl.load(token_idx_ptr + row)
    src_offsets = row * stride_src0 + cols * stride_src1
    out_offsets = token_idx * stride_out0 + cols * stride_out1

    src = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + out_offsets, src, mask=mask)


@triton.jit
def fused_moe_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [E, N, K]
    c_ptr,  # [EM, N]
    topk_weights_ptr,  # [EM]
    sorted_token_ids_ptr,  # [EM]
    expert_ids_ptr,  # [num_pid_m]
    num_tokens_post_padded_ptr,  # scalar
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < EM, other=num_valid_tokens)
    token_mask = (offs_token_id < EM) & (offs_token < num_valid_tokens)

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert < 0:
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + offs_n[None, :] * stride_bn
        + offs_k[:, None] * stride_bk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token_id,
            mask=offs_token_id < EM,
            other=0.0,
        )
        accumulator *= moe_weight[:, None]

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + offs_token_id[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_token_id[:, None] < EM) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def moe_scatter_add_forward(out: torch.Tensor, token_indices: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if src.numel() == 0:
        return out
    if (not out.is_cuda) or (not src.is_cuda):
        out.index_add_(0, token_indices, src.to(out.dtype))
        return out

    out_contig = out.contiguous()
    src_contig = src.contiguous().to(out_contig.dtype)
    token_indices_contig = token_indices.contiguous()

    n_rows, hidden_dim = src_contig.shape
    block_size = min(1024, triton.next_power_of_2(hidden_dim))
    grid = (n_rows, triton.cdiv(hidden_dim, block_size))

    _moe_scatter_add_kernel[grid](
        src_ptr=src_contig,
        token_idx_ptr=token_indices_contig,
        out_ptr=out_contig,
        stride_src0=src_contig.stride(0),
        stride_src1=src_contig.stride(1),
        stride_out0=out_contig.stride(0),
        stride_out1=out_contig.stride(1),
        hidden_dim=hidden_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out_contig


def _pack_expert_tokens(
    sorted_tokens: torch.Tensor,
    sorted_experts: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_valid_tokens: int,
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_tokens = []
    padded_weights = []
    expert_ids = []

    num_experts = int(sorted_experts.max().item()) + 1 if sorted_experts.numel() > 0 else 0
    for expert_id in range(num_experts):
        mask = sorted_experts == expert_id
        exp_tokens = sorted_tokens[mask]
        exp_weights = sorted_weights[mask]
        if exp_tokens.numel() == 0:
            continue
        num_blocks = triton.cdiv(exp_tokens.numel(), block_size_m)
        target = num_blocks * block_size_m
        pad = target - exp_tokens.numel()
        if pad > 0:
            exp_tokens = torch.cat(
                [
                    exp_tokens,
                    torch.full((pad,), num_valid_tokens, device=exp_tokens.device, dtype=exp_tokens.dtype),
                ],
                dim=0,
            )
            exp_weights = torch.cat(
                [exp_weights, torch.zeros((pad,), device=exp_weights.device, dtype=exp_weights.dtype)],
                dim=0,
            )
        padded_tokens.append(exp_tokens)
        padded_weights.append(exp_weights)
        expert_ids.append(torch.full((num_blocks,), expert_id, device=exp_tokens.device, dtype=torch.int32))

    if not padded_tokens:
        empty_i = sorted_tokens.new_empty((0,))
        empty_w = sorted_weights.new_empty((0,))
        empty_e = torch.empty((0,), device=sorted_tokens.device, dtype=torch.int32)
        return empty_i, empty_w, empty_i, empty_w, empty_e

    padded_tokens = torch.cat(padded_tokens, dim=0)
    padded_weights = torch.cat(padded_weights, dim=0)
    expert_ids = torch.cat(expert_ids, dim=0)

    valid_mask = padded_tokens < num_valid_tokens
    valid_positions = valid_mask.nonzero(as_tuple=False).flatten()

    valid_tokens = padded_tokens[valid_positions]
    valid_weights = padded_weights[valid_positions]
    return padded_tokens, padded_weights, valid_tokens, valid_weights, expert_ids


def _launch_fused_moe(
    a: torch.Tensor,
    b: torch.Tensor,
    padded_tokens: torch.Tensor,
    padded_weights: torch.Tensor,
    expert_ids: torch.Tensor,
    mul_routed_weight: bool,
) -> torch.Tensor:
    em = padded_tokens.numel()
    n = b.shape[1]
    k = b.shape[2]

    c = torch.empty((em, n), device=a.device, dtype=a.dtype)
    num_tokens_post_padded = torch.tensor([em], device=a.device, dtype=torch.int32)

    block_m = 64
    block_n = 64
    block_k = 32
    group_m = 8
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)

    fused_moe_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        topk_weights_ptr=padded_weights,
        sorted_token_ids_ptr=padded_tokens,
        expert_ids_ptr=expert_ids,
        num_tokens_post_padded_ptr=num_tokens_post_padded,
        N=n,
        K=k,
        EM=em,
        num_valid_tokens=a.shape[0],
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_be=b.stride(0),
        stride_bn=b.stride(1),
        stride_bk=b.stride(2),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        num_warps=4,
        num_stages=2,
    )
    return c


def qwen3_moe_triton_forward(
    hidden_states: torch.Tensor,
    gate: nn.Linear,
    experts: nn.ModuleList,
    top_k: int,
    num_experts: int,
) -> torch.Tensor:
    if hidden_states.numel() == 0:
        return hidden_states

    router_logits = gate(hidden_states)
    routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    flat_experts = selected_experts.reshape(-1)
    flat_tokens = (
        torch.arange(hidden_states.shape[0], device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    flat_weights = routing_weights.reshape(-1)

    order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[order]
    sorted_tokens = flat_tokens[order]
    sorted_weights = flat_weights[order]

    if (not hidden_states.is_cuda) or sorted_tokens.numel() == 0:
        # CPU fallback preserves correctness.
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_id in range(num_experts):
            mask = sorted_experts == expert_id
            if not mask.any():
                continue
            token_ids = sorted_tokens[mask]
            w = sorted_weights[mask]
            x = hidden_states.index_select(0, token_ids)
            y = experts[expert_id](x) * w[:, None]
            final_hidden_states.index_add_(0, token_ids, y)
        return final_hidden_states

    block_size_m = 64
    padded_tokens, padded_weights, valid_tokens, _, expert_ids = _pack_expert_tokens(
        sorted_tokens,
        sorted_experts,
        sorted_weights,
        num_valid_tokens=hidden_states.shape[0],
        block_size_m=block_size_m,
    )

    if padded_tokens.numel() == 0:
        return torch.zeros_like(hidden_states)

    up_w = torch.stack([expert.gate_up_proj.weight for expert in experts], dim=0).contiguous()
    down_w = torch.stack([expert.down_proj.weight for expert in experts], dim=0).contiguous()

    up = _launch_fused_moe(
        a=hidden_states,
        b=up_w,
        padded_tokens=padded_tokens,
        padded_weights=padded_weights,
        expert_ids=expert_ids,
        mul_routed_weight=False,
    )

    gate_out, up_out = up.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate_out) * up_out

    down = _launch_fused_moe(
        a=act,
        b=down_w,
        padded_tokens=torch.arange(act.shape[0], device=act.device, dtype=padded_tokens.dtype),
        padded_weights=padded_weights,
        expert_ids=expert_ids,
        mul_routed_weight=True,
    )

    valid_mask = padded_tokens < hidden_states.shape[0]
    valid_down = down[valid_mask]
    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states = moe_scatter_add_forward(final_hidden_states, valid_tokens, valid_down)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(final_hidden_states)

    return final_hidden_states
