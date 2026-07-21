#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void cross_entropy_row_kernel(const float* logits, const int* target, float* partial, int B, int C) {
    int row = blockIdx.x;
    if (row >= B) return;

    int tid = threadIdx.x;
    const float* row_logits = logits + row * C;

    __shared__ float sdata[256];

    float local_max = -FLT_MAX;
    for (int j = tid; j < C; j += blockDim.x) {
        float v = row_logits[j];
        if (v > local_max) local_max = v;
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        local_sum += expf(row_logits[j] - row_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = logf(sdata[0]);
        float loss = row_max + lse - row_logits[target[row]];
        atomicAdd(partial, loss);
    }
}

__global__ void cross_entropy_finalize_kernel(float* loss, int B) {
    loss[0] /= (float)B;
}

extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int threads = 256;
    dim3 blocks(B);
    cross_entropy_row_kernel<<<blocks, threads>>>(logits, target, loss, B, C);
    cross_entropy_finalize_kernel<<<1, 1>>>(loss, B);
    cudaDeviceSynchronize();
}