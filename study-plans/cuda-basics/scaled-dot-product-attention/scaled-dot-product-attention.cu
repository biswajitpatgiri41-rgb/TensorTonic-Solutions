#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += Q[row * D + k] * K[col * D + k];
        }
        scores[row * N + col] = sum / sqrtf((float)D);
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[256];

    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = scores[row * N + j];
        if (v > local_max) local_max = v;
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    float maxv = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(scores[row * N + j] - maxv);
        scores[row * N + j] = e;
        local_sum += e;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    float sum = sdata[0];

    for (int j = tid; j < N; j += blockDim.x) {
        scores[row * N + j] /= sum;
    }
}

__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && d < D) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += attn[row * N + j] * V[j * D + d];
        }
        output[row * D + d] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    softmax_rows_kernel<<<N, 256>>>(scores, N);

    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    cudaDeviceSynchronize();
    cudaFree(scores);
}