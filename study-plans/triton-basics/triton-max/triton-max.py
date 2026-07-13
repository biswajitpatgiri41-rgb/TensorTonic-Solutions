import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    tl.store(out_ptr, m)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)