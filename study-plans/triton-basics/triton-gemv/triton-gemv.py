import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, N, BLOCK_N):
        cols = k + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        a = tl.load(
            a_ptr + rows[:, None] * stride_am + cols[None, :] * stride_an,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        x = tl.load(x_ptr + cols, mask=col_mask, other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)

    tl.store(out_ptr + rows, acc, mask=row_mask)


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )