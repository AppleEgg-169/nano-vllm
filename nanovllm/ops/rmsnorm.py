import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 65536


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    eps,
    stride_x0,
    stride_x1,
    stride_out0,
    stride_out1,
    n_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_col

    x = tl.load(x_ptr + row * stride_x0 + cols * stride_x1, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_col
    inv = tl.rsqrt(var + eps)
    out = (x * inv) * w
    tl.store(out_ptr + row * stride_out0 + cols * stride_out1, out, mask=mask)


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    out_ptr,
    residual_out_ptr,
    eps,
    stride_x0,
    stride_x1,
    stride_res0,
    stride_res1,
    stride_out0,
    stride_out1,
    stride_res_out0,
    stride_res_out1,
    n_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_col

    x = tl.load(x_ptr + row * stride_x0 + cols * stride_x1, mask=mask, other=0.0).to(
        tl.float32
    )
    residual = tl.load(
        residual_ptr + row * stride_res0 + cols * stride_res1, mask=mask, other=0.0
    ).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    summed = x + residual
    var = tl.sum(summed * summed, axis=0) / n_col
    inv = tl.rsqrt(var + eps)
    out = (summed * inv) * w

    tl.store(out_ptr + row * stride_out0 + cols * stride_out1, out, mask=mask)
    tl.store(
        residual_out_ptr + row * stride_res_out0 + cols * stride_res_out1,
        summed,
        mask=mask,
    )


def _ensure_forward_only(*tensors: torch.Tensor) -> None:
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError(
            "Triton RMSNorm is forward-only in nano-vllm and does not support backward."
        )


def _check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(
            f"Triton RMSNorm requires CUDA tensors only. `{name}` is on {tensor.device}."
        )


def _check_input(x: torch.Tensor, weight: torch.Tensor) -> None:
    _check_cuda_tensor("x", x)
    _check_cuda_tensor("weight", weight)
    if x.shape[-1] != weight.numel():
        raise RuntimeError(
            "Triton RMSNorm expects `weight.numel() == x.shape[-1]`, got "
            f"{weight.numel()} and {x.shape[-1]}."
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            "Triton RMSNorm supports x dtype in {float16, bfloat16, float32}, got "
            f"{x.dtype}."
        )


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


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    _ensure_forward_only(x, weight)
    _check_input(x, weight)

    x = x.contiguous()
    weight = weight.contiguous()
    n_col = x.shape[-1]
    x_2d = x.view(-1, n_col)
    n_row = x_2d.shape[0]
    if n_row == 0:
        return torch.empty_like(x)

    out_2d = torch.empty_like(x_2d)
    block_size, num_warps = _select_launch_config(n_col)
    _rms_norm_kernel[(n_row,)](
        x_2d,
        weight,
        out_2d,
        eps,
        x_2d.stride(0),
        x_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        n_col,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.view_as(x)


def add_rms_norm_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_forward_only(x, residual, weight)
    _check_input(x, weight)
    _check_cuda_tensor("residual", residual)
    if x.shape != residual.shape:
        raise RuntimeError(
            "Triton add+RMSNorm expects x and residual with same shape, got "
            f"{x.shape} and {residual.shape}."
        )

    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    n_col = x.shape[-1]
    x_2d = x.view(-1, n_col)
    residual_2d = residual.view(-1, n_col)
    n_row = x_2d.shape[0]
    if n_row == 0:
        empty = torch.empty_like(x)
        return empty, empty

    out_2d = torch.empty_like(x_2d)
    residual_out_2d = torch.empty_like(residual_2d)
    block_size, num_warps = _select_launch_config(n_col)
    _add_rms_norm_kernel[(n_row,)](
        x_2d,
        residual_2d,
        weight,
        out_2d,
        residual_out_2d,
        eps,
        x_2d.stride(0),
        x_2d.stride(1),
        residual_2d.stride(0),
        residual_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        residual_out_2d.stride(0),
        residual_out_2d.stride(1),
        n_col,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.view_as(x), residual_out_2d.view_as(residual)
