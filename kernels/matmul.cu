#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C, 
                            int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}