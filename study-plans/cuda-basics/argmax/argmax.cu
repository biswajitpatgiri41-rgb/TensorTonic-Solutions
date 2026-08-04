#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float svals[256];
    __shared__ int sidxs[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float bestVal = -FLT_MAX;
    int bestIdx = -1;

    if (gid < N) {
        bestVal = input[gid];
        bestIdx = gid;
    }

    svals[tid] = bestVal;
    sidxs[tid] = bestIdx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v1 = svals[tid];
            float v2 = svals[tid + stride];
            int i1 = sidxs[tid];
            int i2 = sidxs[tid + stride];

            if (v2 > v1 || (v2 == v1 && i2 < i1)) {
                svals[tid] = v2;
                sidxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = svals[0];
        block_idxs[blockIdx.x] = sidxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float svals[256];
    __shared__ int sidxs[256];

    int tid = threadIdx.x;

    float bestVal = -FLT_MAX;
    int bestIdx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        int idx = block_idxs[i];

        if (v > bestVal || (v == bestVal && idx < bestIdx) || bestIdx == -1) {
            bestVal = v;
            bestIdx = idx;
        }
    }

    svals[tid] = bestVal;
    sidxs[tid] = bestIdx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v1 = svals[tid];
            float v2 = svals[tid + stride];
            int i1 = sidxs[tid];
            int i2 = sidxs[tid + stride];

            if (v2 > v1 || (v2 == v1 && i2 < i1)) {
                svals[tid] = v2;
                sidxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0)
        result[0] = sidxs[0];
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals;
    int* block_idxs;

    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);

    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}