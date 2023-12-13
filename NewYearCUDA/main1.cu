#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#include "cuda_include.cuh"

// CUDA kernel for horizontal mirroring
__global__ void robst(const uchar *inputImage, uchar *outputImage, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    const auto channels = 1;

    if (x < width && y < height && (x - 1) > 0 && (y - 1) > 0) {

        int g_x =
                -1 * inputImage[GET_INDEX_ON_CUDA(x - 1, y - 1, width, channels)] +
                -1 * inputImage[GET_INDEX_ON_CUDA(x - 1, y, width, channels)] +
                1 * inputImage[GET_INDEX_ON_CUDA(x, y - 1, width, channels)] +
                1 * inputImage[GET_INDEX_ON_CUDA(x, y, width, channels)];

        int g_y =
                -1 * inputImage[GET_INDEX_ON_CUDA(x - 1, y - 1, width, channels)] +
                1 * inputImage[GET_INDEX_ON_CUDA(x - 1, y, width, channels)] +
                -1 * inputImage[GET_INDEX_ON_CUDA(x, y - 1, width, channels)] +
                1 * inputImage[GET_INDEX_ON_CUDA(x, y, width, channels)];

        auto g = static_cast<uchar>(sqrtf(g_x * g_x + g_y * g_y));

        outputImage[GET_INDEX_ON_CUDA(x, y, width, 1)] = g;

    }
}

uchar findMax(const uchar *array, int size) {
    if (size <= 0) {
        // 处理数组为空的情况
        std::cerr << "Error: Array is empty." << std::endl;
        return 0;
    }

    uchar maxVal = array[0]; // 初始化为数组的第一个元素

    for (int i = 1; i < size; ++i) {
        if (array[i] > maxVal) {
            maxVal = array[i];
        }
    }

    return static_cast<uchar>(maxVal); // 将 unsigned char 转换为 int 返回
}

int main() {
    // Read the image using OpenCV
    cv::Mat inputImage = cv::imread("../test3.bmp", cv::IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        std::cerr << "Failed to read the image." << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory on the host
    uchar *hostInputImage = (uchar *) inputImage.ptr();
    uchar *hostOutputImage = new uchar[width * height];

    // Allocate memory on the device
    uchar *deviceInputImage;
    uchar *deviceOutputImage;

    cudaMalloc((void **) &deviceInputImage, width * height * sizeof(uchar));
    cudaMalloc((void **) &deviceOutputImage, width * height * sizeof(uchar));

    // Copy the input image to the device
    cudaMemcpy(deviceInputImage, hostInputImage, width * height * sizeof(uchar), cudaMemcpyHostToDevice);

    // Specify block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    robst<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height);

    // Copy the result back to the host
    cudaMemcpy(hostOutputImage, deviceOutputImage, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

//    auto max = findMax(hostOutputImage, width * height);

    uchar max = 0;
    for (int i = 0; i < width * height; ++i) {
        if (hostOutputImage[i] > max) {
            max = hostOutputImage[i];
        }
    }
    std::cout << "max: " << max << std::endl;
    for (int i = 0; i < width * height; i++) {
        hostOutputImage[i] = static_cast<uchar>(255.0f * hostOutputImage[i] / max);
    }

    // Create a new OpenCV image with the mirrored data
    cv::Mat outputImage(height, width, CV_8UC1, hostOutputImage);

    // Display the original and mirrored images
    cv::imshow("Original Image", inputImage);
    cv::imshow("Output Image", outputImage);
    cv::waitKey(0);

    // Free memory
    delete[] hostOutputImage;
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    return 0;
}
