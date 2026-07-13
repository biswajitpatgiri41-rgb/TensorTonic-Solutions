import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * x_row_stride
    out_row_ptr = out_ptr + row * out_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_row_ptr + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_row_ptr + cols, y, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x,
        out,
        x.stride(0),
        out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )