#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>
using namespace std::chrono;

#define EPSILON 0.001

__global__ void naiveMatrixMul(float* A, float* B, float* C, int dim) {
    // We are expected to calculate one particular element of C
    // The exact element (i,j) is given by (blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < dim && j < dim) {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            sum += A[i * dim + k] * B[k * dim + j];
        } 
        C[i * dim + j] = sum;
    }
}

__global__ void coalescedMatrixMul(float* A, float* B, float* C, int dim) {
    // We are expected to calculate one particular element of C
    // The exact element (i,j) is given by (blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim && j < dim) {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            sum += A[i * dim + k] * B[k * dim + j];
        } 
        C[i * dim + j] = sum;
    }
}

__global__ void dummy(float *X, float *Y, int N) {
    float x = Y[threadIdx.y * N + threadIdx.x];
    x += 1.0f;
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
    dim3 g(32, 32);
    dim3 b(32, 32);
    dummy<<<g,b>>>(A,A,size);
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
    if (mode & 4) {
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
        coalescedMatrixMul<<<grid , block>>>(AGPU,BGPU,CGPU,size);
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
        naiveMatrixMul<<<grid , block>>>(AGPU,BGPU,CGPU,size);
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
    if (mode & 8) {
        auto prev = high_resolution_clock::now();
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        float *AGPU = nullptr;
        float *BGPU = nullptr;
        float *CGPU = nullptr;
        cudaMalloc(&AGPU, sizeof(float) * size * size);
        cudaMalloc(&BGPU, sizeof(float) * size * size);
        cudaMalloc(&CGPU, sizeof(float) * size * size);
        cudaMemcpy(AGPU, A, sizeof(float) * size * size, cudaMemcpyDefault);
        cudaMemcpy(BGPU, B, sizeof(float) * size * size, cudaMemcpyDefault);
        cudaMemset(CGPU, 0, sizeof(float) * size * size);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,BGPU, size, AGPU, size, &beta, CGPU, size);
        cublasDestroy(handle);
        cudaMemcpy(CCpy, CGPU, sizeof(float) * size * size, cudaMemcpyDefault);
        cudaFree(AGPU);
        cudaFree(BGPU);
        cudaFree(CGPU);
        auto curr = high_resolution_clock::now();
        if (mode & 1) {
            for (int i=0; i < size; i++) {
                for (int j=0; j < size; j++) {
                    if (std::fabs(CCpy[i * size + j] - C[i * size + j]) > EPSILON) {
                        std::cout << "Incorrect result of matmul at indices: (" << i << ", " << j << ")" << std::endl;
                        std::cout << "Expected: " << C[i * size + j] << ", Got: " << CCpy[i * size + j] << std::endl;
                        return 1;
                    }
                }
            }
        }
        auto duration = duration_cast<microseconds>(curr - prev);
        std::cout << "[CUBLAS] Time Taken: " << duration.count() << " microseconds" << std::endl;
    }
    delete []A;
    delete []B;
    delete []C;
    delete []CCpy;
}