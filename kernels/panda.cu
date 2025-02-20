#include <cstdint>

#define TILE_SIZE 32
#define BITS_PER_INT 32

__global__ void spmpgemm_kernel(
    const int8_t* __restrict__ A,
    const uint32_t* __restrict__ B_bits,
    int32_t* __restrict__ C,
    int M, int N, int K,
    int bit_shift
) {
    // Tile storage in shared memory
    __shared__ int8_t Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ uint32_t Bshared[TILE_SIZE][TILE_SIZE / BITS_PER_INT + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int32_t sum = 0;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // Load A tile
        int a_row = row;
        int a_col = k_tile + threadIdx.x;
        if (a_row < M && a_col < K) {
            Ashared[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            Ashared[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tile (bit plane)
        int b_row = col;
        int b_col = k_tile + threadIdx.y;
        if (b_row < N && b_col < K) {
            int word_idx = b_col / BITS_PER_INT;
            int bit_idx = b_col % BITS_PER_INT;
            uint32_t word = B_bits[b_row * ((K + BITS_PER_INT - 1) / BITS_PER_INT) + word_idx];
            Bshared[threadIdx.y][threadIdx.x] = (word >> bit_idx) & 1;
        } else {
            Bshared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute dot product for the current tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            int8_t a = Ashared[threadIdx.y][i];
            uint32_t b = Bshared[i][threadIdx.x];
            sum += a * static_cast<int32_t>(b);
        }

        __syncthreads();
    }

    // Apply bit shift and accumulate to output
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum << bit_shift);
    }
}