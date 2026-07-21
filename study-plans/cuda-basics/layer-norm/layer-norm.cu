#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* sum = shared;
    float* sumsq = shared + blockDim.x;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    int base = row * N;

    for (int j = tid; j < N; j += blockDim.x) {
        float v = input[base + j];
        local_sum += v;
        local_sumsq += v * v;
    }

    sum[tid] = local_sum;
    sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum[tid] += sum[tid + stride];
            sumsq[tid] += sumsq[tid + stride];
        }
        __syncthreads();
    }

    float mean = sum[0] / (float)N;
    float var = sumsq[0] / (float)N - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int j = tid; j < N; j += blockDim.x) {
        float x = input[base + j];
        output[base + j] = (x - mean) * inv_std * gamma[j] + beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    size_t shared_mem = 2 * threads * sizeof(float);
    layer_norm_kernel<<<blocks, threads, shared_mem>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}