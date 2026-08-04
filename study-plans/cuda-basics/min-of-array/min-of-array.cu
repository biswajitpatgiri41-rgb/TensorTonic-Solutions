#include <cuda_runtime.h>
#include <float.h>

__device__ float atomicMinFloat(float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int;
    int assumed;

    while (__int_as_float(old) > value) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (assumed == old) break;
    }

    return __int_as_float(old);
}

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* result, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float v = (idx < N) ? input[idx] : FLT_MAX;
    sdata[tid] = v;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinFloat(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, result, N);
    cudaDeviceSynchronize();
}