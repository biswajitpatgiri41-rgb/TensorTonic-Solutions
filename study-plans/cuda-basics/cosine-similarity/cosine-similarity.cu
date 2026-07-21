#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    __shared__ float sDot[256];
    __shared__ float sA2[256];
    __shared__ float sB2[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float dot = 0.0f;
    float a2 = 0.0f;
    float b2 = 0.0f;

    for (int i = idx; i < N; i += stride) {
        float a = A[i];
        float b = B[i];
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
    }

    sDot[tid] = dot;
    sA2[tid] = a2;
    sB2[tid] = b2;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sDot[tid] += sDot[tid + offset];
            sA2[tid] += sA2[tid + offset];
            sB2[tid] += sB2[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&scratch[0], sDot[0]);
        atomicAdd(&scratch[1], sA2[0]);
        atomicAdd(&scratch[2], sB2[0]);
    }
}

__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    float dot = scratch[0];
    float a2 = scratch[1];
    float b2 = scratch[2];
    result[0] = dot / (sqrtf(a2) * sqrtf(b2));
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float* scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);

    cudaDeviceSynchronize();
    cudaFree(scratch);
}