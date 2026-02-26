import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, stride_x, stride_y, n_row, n_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0)
    for row in range(pid, n_row, row_stride):
        cols = tl.arange(0,BLOCK_SIZE)
        mask = offsets < cols
        offsets = row * stride_x + stride_y * cols
        input = tl.load(input_ptr + offsets, mask=mask,other=-float('inf'))
        input_max = tl.max(input,axis=0)
        input = tl.exp(input - input_max)
        input_sum = tl.sum(input,axis=0)
        out = input / input_sum
        tl.store(output_ptr + offsets, out, mask=mask)

def softmax(x: torch.tensor):
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    grid = lambda meta : (meta["BLOCK_SIZE"],)
    softmax[grid](x,x,x.stride(0),x.stride(1),x.shape[0],x.shape[1])