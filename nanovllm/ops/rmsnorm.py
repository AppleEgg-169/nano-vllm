import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 65536


@triton.jit
def _rmsnorm_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    eps,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    n_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_col

    offsets = row * stride_x0 + cols * stride_x1
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(ss / n_col + eps)
    y = (x * inv_rms) * w

    out_offsets = row * stride_y0 + cols * stride_y1
    tl.store(output_ptr + out_offsets, y, mask=mask)


@triton.jit
def _add_rmsnorm_kernel(
    input_ptr,
    output_ptr,
    residual_ptr,
    residual_out_ptr,
    weight_ptr,
    eps,
    stride_x0,
    stride_x1,
    stride_res0,
    stride_res1,
    stride_y0,
    stride_y1,
    stride_res_out0,
    stride_res_out1,
    n_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_col

    in_offsets = row * stride_x0 + cols * stride_x1
    res_offsets = row * stride_res0 + cols * stride_res1
    out_offsets = row * stride_y0 + cols * stride_y1
    res_out_offsets = row * stride_res_out0 + cols * stride_res_out1

    x = tl.load(input_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + res_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    summed = x + residual
    ss = tl.sum(summed * summed, axis=0)
    inv_rms = tl.rsqrt(ss / n_col + eps)
    y = (summed * inv_rms) * w

    tl.store(residual_out_ptr + res_out_offsets, summed, mask=mask)
    tl.store(output_ptr + out_offsets, y, mask=mask)


def _select_launch_config(n_col: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(n_col)
    if block_size > _MAX_BLOCK_SIZE:
        raise RuntimeError(
            "Triton RMSNorm unsupported hidden size: "
            f"{n_col}, block size {block_size} exceeds {_MAX_BLOCK_SIZE}."
        )
    if block_size <= 1024:
        num_warps = 4
    elif block_size <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    return block_size, num_warps


def _check_input(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError(f"Triton RMSNorm requires CUDA tensor `x`, got {x.device}.")
    if not weight.is_cuda:
        raise RuntimeError(
            f"Triton RMSNorm requires CUDA tensor `weight`, got {weight.device}."
        )
    if x.shape[-1] != weight.numel():
        raise RuntimeError(
            "Triton RMSNorm expects `weight.numel() == x.shape[-1]`, got "
            f"{weight.numel()} and {x.shape[-1]}."
        )


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    _check_input(x, weight)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError("Triton RMSNorm is forward-only in nano-vllm.")

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    n_col = x_contig.shape[-1]
    x_2d = x_contig.view(-1, n_col)
    out_2d = torch.empty_like(x_2d)
    n_row = x_2d.shape[0]
    if n_row == 0:
        return torch.empty_like(x_contig)

    block_size, num_warps = _select_launch_config(n_col)
    _rmsnorm_kernel[(n_row,)](
        input_ptr=x_2d,
        output_ptr=out_2d,
        weight_ptr=w_contig,
        eps=eps,
        stride_x0=x_2d.stride(0),
        stride_x1=x_2d.stride(1),
        stride_y0=out_2d.stride(0),
        stride_y1=out_2d.stride(1),
        n_col=n_col,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.view_as(x_contig)


def add_rms_norm_forward(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_input(x, weight)
    if not residual.is_cuda:
        raise RuntimeError(
            f"Triton add+RMSNorm requires CUDA tensor `residual`, got {residual.device}."
        )
    if x.shape != residual.shape:
        raise RuntimeError(
            "Triton add+RMSNorm expects x and residual with same shape, got "
            f"{x.shape} and {residual.shape}."
        )
    if torch.is_grad_enabled() and (
        x.requires_grad or residual.requires_grad or weight.requires_grad
    ):
        raise RuntimeError("Triton add+RMSNorm is forward-only in nano-vllm.")

    x_contig = x.contiguous()
    residual_contig = residual.contiguous()
    w_contig = weight.contiguous()
    n_col = x_contig.shape[-1]
    x_2d = x_contig.view(-1, n_col)
    residual_2d = residual_contig.view(-1, n_col)
    out_2d = torch.empty_like(x_2d)
    residual_out_2d = torch.empty_like(residual_2d)
    n_row = x_2d.shape[0]
    if n_row == 0:
        empty = torch.empty_like(x_contig)
        return empty, empty

    block_size, num_warps = _select_launch_config(n_col)
    _add_rmsnorm_kernel[(n_row,)](
        input_ptr=x_2d,
        output_ptr=out_2d,
        residual_ptr=residual_2d,
        residual_out_ptr=residual_out_2d,
        weight_ptr=w_contig,
        eps=eps,
        stride_x0=x_2d.stride(0),
        stride_x1=x_2d.stride(1),
        stride_res0=residual_2d.stride(0),
        stride_res1=residual_2d.stride(1),
        stride_y0=out_2d.stride(0),
        stride_y1=out_2d.stride(1),
        stride_res_out0=residual_out_2d.stride(0),
        stride_res_out1=residual_out_2d.stride(1),
        n_col=n_col,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.view_as(x_contig), residual_out_2d.view_as(residual_contig)
