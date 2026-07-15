import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * x_row_stride

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(row_ptr + offs, mask=mask, other=-float("inf"))
    row_max = tl.max(x, axis=0)

    exp_x = tl.exp(x - row_max)
    exp_x = tl.where(mask, exp_x, 0.0)
    row_sum = tl.sum(exp_x, axis=0)

    lse = row_max + tl.log(row_sum)
    tl.store(out_ptr + pid, lse)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    logsumexp_kernel[(M,)](
        x_ptr=x,
        out_ptr=out,
        x_row_stride=x.stride(0),
        n_cols=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )