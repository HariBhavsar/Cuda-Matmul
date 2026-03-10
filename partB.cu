#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>
using namespace std::chrono;

#define EPSILON 0.001
#define INDEX(I,J,SZ) ((((I) * (SZ)) + (J)))
const int tileSize = 32;

__global__ void tileOne(float* A, float* B, float* C, int dim) {
    __shared__ float tileA[tileSize * tileSize];
    __shared__ float tileB[tileSize * tileSize];
    /*
    Procedure:
    1. Given thread with (bX, bY, tX, tY) is responsible for computing C[INDEX(bY * dimY + tY, bX * dimX + tX, dim)]
    2. For tile_id in dim/tileSize:
        2.1: Load tile of A and B into shared memory, each thread loads A[INDEX(bY * dimY + tY, tile_id * tileSize + tX, dim)] and B[INDEX(tile_id * tileSize + tY, bX * dimX + tX, dim)]
        2.2: Each thread does for k in (0, tileSize):
            C[INDEX(bY * dimY + tY, bX * dimX + tX, dim)] += A[INDEX(bY * dimY + tY, tile_id * tileSize + k, dim)] * B[INDEX(tile_id * tileSize + k, bX * dimX + tX, dim)]
    */

   float localSum = 0.0f;
    for (int tileId = 0; tileId < (dim/tileSize); tileId++) {
        tileA[INDEX(threadIdx.y, threadIdx.x, tileSize)] = A[INDEX(blockIdx.y * blockDim.y + threadIdx.y, tileId * tileSize + threadIdx.x, dim)];
        tileB[INDEX(threadIdx.y, threadIdx.x, tileSize)] = B[INDEX(tileId * tileSize + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, dim)];
        __syncthreads();
        for (int k=0; k < tileSize; k++) {
            localSum += tileA[INDEX(threadIdx.y, k, tileSize)] * tileB[INDEX(k, threadIdx.x, tileSize)];
        }
        __syncthreads();
    }
    C[INDEX(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x, dim)] = localSum;

}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./partA <Matrix Size> <Mode: 0 -> CPU, 1 -> GPU, 2 -> Both>" << std::endl;
        return 1;       
    }   
    int size = std::stoi(argv[1]);
    int mode = std::stoi(argv[2]);
    float *A = new float[size * size];
    float *B = new float[size * size];
    float *C = new float[size * size];
    float *CCpy = new float[size * size];
    for (int i=0; i < size; i++) {
        for (int j=0; j < size; j++) {
            A[i * size + j] = (((float)(i+j))/((float)(size * size)));
            B[i * size + j] = (((float)(i-j))/((float)(size * size)));
            C[i * size + j] = 0.0f;
            CCpy[i * size + j] = 0.0f;
        }
    }
    std::cout << "Matrices setup successfully" << std::endl;
    // First, CPU based matrix multiplication
    if (mode & 1) {
        auto prev = high_resolution_clock::now();
        for (int i=0; i < size; i++) {
            for (int j=0; j < size; j++) {
                for (int k=0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k*size + j];
                }
            }
        }
        auto curr = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(curr - prev);
        std::cout << "[CPU] Time taken: " << duration.count() << " microseconds" << std::endl;
    }
    if (mode & 2) {
        // GPU based matrix multiplication
        assert(size % 1024 == 0);
        auto prev = high_resolution_clock::now();
        float *AGPU = nullptr;
        float *BGPU = nullptr;
        float *CGPU = nullptr;
        cudaMalloc(&AGPU, sizeof(float) * size * size);
        cudaMalloc(&BGPU, sizeof(float) * size * size);
        cudaMalloc(&CGPU, sizeof(float) * size * size);
        cudaMemcpy(AGPU, A, sizeof(float) * size * size, cudaMemcpyDefault);
        cudaMemcpy(BGPU, B, sizeof(float) * size * size, cudaMemcpyDefault);
        cudaMemset(CGPU, 0, sizeof(float) * size * size);
        dim3 block(32,32);
        dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);
        tileOne<<<grid , block>>>(AGPU,BGPU,CGPU,size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(CCpy, CGPU, sizeof(float) * size * size, cudaMemcpyDefault);
        auto curr = high_resolution_clock::now();
        if (mode & 1) {
            for (int i=0; i < size; i++) {
                for (int j=0; j < size; j++) {
                    if (std::fabs(CCpy[i * size + j] - C[i * size + j])/C[i * size + j] > EPSILON) {
                        std::cout << "Incorrect result of matmul at indices: (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Expected: " << C[i * size + j] << ", Got: " << CCpy[i * size + j] << std::endl;
                        return 1;
                    }
                }
            }
        }
        cudaFree(AGPU);
        cudaFree(BGPU);
        cudaFree(CGPU);
        auto duration = duration_cast<microseconds>(curr - prev);
        std::cout << "[GPU] Time Taken: " << duration.count() << " microseconds" << std::endl;
    }
}