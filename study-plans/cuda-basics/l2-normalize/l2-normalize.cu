#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float v = input[i];
        sum += v * v;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *sumv = sdata[0];
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float inv = rsqrtf(*sumv);
        output[idx] = input[idx] * inv;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}