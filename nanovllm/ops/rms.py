import torch
import triton
import triton.language as tl        
    
@triton.jit
def rmsnorm_kernel(
    input_ptr,
    output_ptr,          
    weight_ptr,         
    eps,
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    n_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0,BLOCK_SIZE)
    mask = cols < n_col
    offsets_input = pid * stride_x0 + cols * stride_x1
    offsets_out  = pid * stride_y0 + cols * stride_y1

    x = tl.load(input_ptr + offsets_input,mask=mask,other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols,mask=mask,other=0.0).to(tl.float32)

    sum = tl.sum(x * x,axis=0) / n_col
    z = tl.rsqrt(sum - eps)
    out = x * z * w

    tl.load(output_ptr + offsets_out,out,mask=mask)
    


def rms_triton(x: torch.Tensor, weight, eps):
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    output = torch.empty_like(x)
    rmsnorm_kernel[(128,)](
        input_ptr=x,
        output_ptr=output,
        weight_ptr=weight,
        eps=eps,
        stride_x0=x.stride(0),
        stride_x1=x.stride(1),
        stride_y0=output.stride(0),
        stride_y1=output.stride(1),
        n_row=x.shape[0],
        n_col=x.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@triton.jit
def add_rmsnorm_kernel(
    input_ptr,
    output_ptr,
    residual_ptr,
    residual_out_ptr,          
    weight_ptr,         
    eps,
    stride_x0, stride_x1,
    stride_res0, stride_res1,
    stride_y0, stride_y1,
    stride_res_out0, stride_res_out1,
    n_row, n_col,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_stride = tl.num_programs(0)

    while row < n_row:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_col
        offsets_in = row * stride_x0 + cols * stride_x1
        offsets_out = row * stride_y0 + cols * stride_y1
        offsets_res = row * stride_res0 + cols * stride_res1
        offsets_resout = row * stride_res_out0 + cols * stride_res_out1

        x = tl.load(input_ptr + offsets_in, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets_res,mask=mask,other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x += residual
        ss = tl.sum(x * x, axis=0)
        inv_rms = tl.rsqrt(ss / n_col + eps)
        y = (x * inv_rms) * w

        tl.store(residual_out_ptr + offsets_resout,x,mask=mask)
        tl.store(output_ptr + offsets_out, y, mask=mask)
        row += row_stride


def add_rms_triton(x: torch.Tensor, residual, weight, eps):
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    output = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    add_rmsnorm_kernel[(128,)](
        input_ptr=x,
        residual_ptr=residual,
        residual_out_ptr=residual_out,
        output_ptr=output,
        weight_ptr=weight,
        eps=eps,
        stride_x0=x.stride(0),
        stride_x1=x.stride(1),
        stride_res0=residual.stride(0),
        stride_res1=residual.stride(1),
        stride_res_out0=residual_out.stride(0),
        stride_res_out1=residual_out.stride(1),
        stride_y0=output.stride(0),
        stride_y1=output.stride(1),
        n_row=x.shape[0],
        n_col=x.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output, residual_out


