#include <iostream>
#include <cmath>
//#include<cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "cuda_include.cuh"

// CUDA kernel for convolution
__global__ void
convolutionKernel(const uchar *input, uchar *result, const float *kernel, int width, int height, int kernelSize) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float value = 0.0f;

        for (int k_i = 0; k_i < kernelSize; ++k_i) {
            for (int k_j = 0; k_j < kernelSize; ++k_j) {
                int y = i + k_i - kernelSize / 2;
                int x = j + k_j - kernelSize / 2;

                if (y >= 0 && y < height && x >= 0 && x < width) {
                    value += input[y * width + x] * kernel[k_i * kernelSize + k_j];
                }
            }
        }

        result[i * width + j] = static_cast<uchar>(value);
    }
}

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("../test3.bmp", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int size = width * height * sizeof(uchar);

    // 定义两个卷积核
    const float kernel_x[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    const float kernel_y[] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
    const float kernel_laplacian[] = {0,1,0, 1,-4,1, 0,1,0};
    const int kernelSize = 3;

    // 在设备上分配内存
    uchar *d_input;
    uchar *d_result_x;
    uchar *d_result_y;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_result_x, size);
    cudaMalloc(&d_result_y, size);

    // 将输入图像复制到设备
    cudaMemcpy(d_input, image.data, size, cudaMemcpyHostToDevice);

    // 将卷积核复制到设备
    float *d_kernel_x;
    float *d_kernel_y;
    cudaMalloc(&d_kernel_x, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_kernel_y, kernelSize * kernelSize * sizeof(float));
//    cudaMemcpy(d_kernel_x, kernel_x, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_y, kernel_y, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_kernel_x, kernel_laplacian, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // 定义块大小和网格大小
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用卷积核函数
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_result_x, d_kernel_x, width, height, kernelSize);
//    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_result_x, d_kernel_x, width, height, kernelSize);

//    convolutionKernel<<<gridSize, blockSize>>>(d_result_x, d_result_y, d_kernel_y, width, height, kernelSize);

    // 将结果从设备复制回主机
    uchar *result_x = new uchar[size];
    uchar *result_y = new uchar[size];
    cudaMemcpy(result_x, d_result_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_y, d_result_y, size, cudaMemcpyDeviceToHost);

//    // 计算最终结果
//    uchar *final_result = new uchar[size];
//    for (int i = 0; i < width * height; ++i) {
//        final_result[i] = static_cast<uchar>(sqrt(result_x[i] * result_x[i] + result_y[i] * result_y[i]));
//    }

    // 将结果保存到图像
    cv::Mat outputImage(height, width, CV_8UC1, result_x);
//    cv::imwrite("output.png", outputImage);
    cv::imshow("Image", image);
    cv::imshow("Output", outputImage);
    cv::waitKey(0);

    // 释放内存
    delete[] result_x;
    delete[] result_y;
//    delete[] final_result;
    cudaFree(d_input);
    cudaFree(d_result_x);
    cudaFree(d_result_y);
    cudaFree(d_kernel_x);
    cudaFree(d_kernel_y);

    return 0;
}
