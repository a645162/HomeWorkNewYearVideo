#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// CUDA kernel function to composite src2 onto dst at specified position after placing src1 on dst
__global__ void blendImages(const uchar* src1, const uchar* src2, uchar* dst, int width1, int height1, int channels1, int width2, int height2, int channels2, int posX, int posY, float opacity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width1 && y < height1)
    {
        int index1 = (y * width1 + x) * channels1;
        int index2 = ((y - posY) * width2 + (x - posX)) * channels2;

        dst[index1] = (src1[index1] * (1 - opacity)) + (src2[index2] * opacity);
        dst[index1 + 1] = (src1[index1 + 1] * (1 - opacity)) + (src2[index2 + 1] * opacity);
        dst[index1 + 2] = (src1[index1 + 2] * (1 - opacity)) + (src2[index2 + 2] * opacity);
    }
}

int main()
{
    // Read input images
    cv::Mat image1 = cv::imread("image1.jpg", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread("image2.png", cv::IMREAD_COLOR);

    cv::imwrite("image1.jpg", image1);
    cv::imwrite("image1.png", image1);
    cv::imwrite("image2.jpg", image2);
    cv::imwrite("image2.png", image2);

    int width1 = image1.cols;
    int height1 = image1.rows;
    int channels1 = image1.channels();

    int width2 = image2.cols;
    int height2 = image2.rows;
    int channels2 = image2.channels();

    // Create and configure CUDA device memory
    uchar* devSrc1, * devSrc2, * devDst;
    cudaMalloc((void**)&devSrc1, width1 * height1 * channels1);
    cudaMalloc((void**)&devSrc2, width2 * height2 * channels2);
    cudaMalloc((void**)&devDst, width1 * height1 * channels1);

    // Copy data from Mat to CUDA device memory
    cudaMemcpy(devSrc1, image1.data, width1 * height1 * channels1, cudaMemcpyHostToDevice);
    cudaMemcpy(devSrc2, image2.data, width2 * height2 * channels2, cudaMemcpyHostToDevice);

    // Define thread block and grid size for CUDA kernel function
    dim3 blockSize(16, 16);
    dim3 gridSize((width1 + blockSize.x - 1) / blockSize.x, (height1 + blockSize.y - 1) / blockSize.y);

    // Set the position and opacity for compositing
    int posX = 100; // X coordinate of the compositing position
    int posY = 50;  // Y coordinate of the compositing position
    float opacity = 0.5f; // Opacity for compositing

    // Execute the CUDA kernel function
    blendImages<<<gridSize, blockSize>>>(devSrc1, devSrc2, devDst, width1, height1, channels1, width2, height2, channels2, posX, posY, opacity);

    // Copy the result from CUDA device memory back to Mat
    cv::Mat result(height1, width1, CV_8UC3);
    cudaMemcpy(result.data, devDst, width1 * height1 * channels1, cudaMemcpyDeviceToHost);

    // Free CUDA device memory
    cudaFree(devSrc1);
    cudaFree(devSrc2);
    cudaFree(devDst);

    // Display the composited result
    cv::imshow("Blended Image", result);
    cv::waitKey(0);

    return 0;
}