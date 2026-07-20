#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outH && j < outW) {
        float mx = -FLT_MAX;
        int baseRow = i * sH;
        int baseCol = j * sW;

        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                mx = fmaxf(mx, input[(baseRow + a) * W + (baseCol + b)]);
            }
        }

        output[i * outW + j] = mx;
    }
}

extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);

    max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);
    cudaDeviceSynchronize();
}