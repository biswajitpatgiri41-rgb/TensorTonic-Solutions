#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    if (blockIdx.x != 0) return;

    __shared__ float shared[256];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float localMax = -FLT_MAX;
    for (int i = tid; i < N; i += stride) {
        if (input[i] > localMax) localMax = input[i];
    }

    shared[tid] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) shared[tid] = shared[tid + s];
        }
        __syncthreads();
    }

    float maxVal = shared[0];

    float localSum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        localSum += expf(input[i] - maxVal);
    }

    shared[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float sumVal = shared[0];

    for (int i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - maxVal) / sumVal;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}