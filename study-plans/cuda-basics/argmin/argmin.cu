#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>

__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float svals[256];
    __shared__ int sidxs[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = FLT_MAX;
    int idx = INT_MAX;

    if (gid < N) {
        val = input[gid];
        idx = gid;
    }

    svals[tid] = val;
    sidxs[tid] = idx;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float v = svals[tid + offset];
            int i = sidxs[tid + offset];
            if (v < svals[tid] || (v == svals[tid] && i < sidxs[tid])) {
                svals[tid] = v;
                sidxs[tid] = i;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = svals[0];
        block_idxs[blockIdx.x] = sidxs[0];
    }
}

__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float svals[256];
    __shared__ int sidxs[256];

    int tid = threadIdx.x;

    float bestVal = FLT_MAX;
    int bestIdx = INT_MAX;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        int idx = block_idxs[i];
        if (v < bestVal || (v == bestVal && idx < bestIdx)) {
            bestVal = v;
            bestIdx = idx;
        }
    }

    svals[tid] = bestVal;
    sidxs[tid] = bestIdx;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float v = svals[tid + offset];
            int i = sidxs[tid + offset];
            if (v < svals[tid] || (v == svals[tid] && i < sidxs[tid])) {
                svals[tid] = v;
                sidxs[tid] = i;
            }
        }
        __syncthreads();
    }

    if (tid == 0)
        result[0] = sidxs[0];
}

extern "C" void solve(const float* input, int* result, int N) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals;
    int* block_idxs;

    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);

    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}