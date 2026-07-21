#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    __shared__ float sdata[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    int base = row * N;

    for (int j = tid; j < N; j += blockDim.x) {
        float v = input[base + j];
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

    float inv_rms = rsqrtf(sdata[0] / (float)N + eps);

    for (int j = tid; j < N; j += blockDim.x) {
        output[base + j] = input[base + j] * inv_rms * gamma[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}