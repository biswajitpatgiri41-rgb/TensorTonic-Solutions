#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_l1_kernel(const float* input, float* sum, int N) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float x = 0.0f;
    if (i < N) x = fabsf(input[i]);
    sdata[tid] = x;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void l1_normalize_kernel(const float* input, float* output, const float* sum, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i] / (*sum);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_l1_kernel<<<blocks, threads>>>(input, d_sum, N);
    l1_normalize_kernel<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();

    cudaFree(d_sum);
}