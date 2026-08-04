#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(val, old_val);
        old = atomicCAS(address_as_i, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* result, int N) {
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < N) ? input[idx] : -FLT_MAX;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}