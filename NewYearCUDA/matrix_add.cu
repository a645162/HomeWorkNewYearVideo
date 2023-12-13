#include <stdio.h>
#include <stdlib.h>

#define SIZE 16

// CUDA核函数，用于计算矩阵元素和
__global__ void matrixElementSum(float *matrix, float *result) {
    auto tid = threadIdx.x + threadIdx.y * blockDim.x;
    auto stride = blockDim.x * blockDim.y;

    // 执行块内归约
    for (auto i = tid; i < SIZE * SIZE; i += stride) {
        atomicAdd(result, matrix[i]);
    }
}

// CPU函数，用于计算矩阵元素和
float cpuMatrixElementSum(const float *matrix) {
    float sum = 0;
    for (int i = 0; i < SIZE * SIZE; ++i) {
        sum += matrix[i];
    }
    return sum;
}

int main() {
    float matrix[SIZE][SIZE];

    // Initialize matrix
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            matrix[i][j] = static_cast<float>(i * SIZE + j + 1)/100000;
        }
    }

    // GPU computation
    float *dev_matrix, *dev_result;
    cudaMalloc((void**)&dev_matrix, SIZE * SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));

    cudaMemcpy(dev_matrix, matrix, SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dev_result, 0, sizeof(float));

    dim3 blockDim(4, 4);  // 使用4x4的线程块
    dim3 gridDim(1, 1);

    matrixElementSum<<<gridDim, blockDim>>>(dev_matrix, dev_result);

    float gpuResult;
    cudaMemcpy(&gpuResult, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    // CPU computation
    float cpuResult = cpuMatrixElementSum((float*)matrix);

    // 打印GPU和CPU结果进行比较
    printf("GPU 矩阵元素和: %f\n", gpuResult);
    printf("CPU 矩阵元素和: %f\n", cpuResult);

    cudaFree(dev_matrix);
    cudaFree(dev_result);

    return 0;
}
