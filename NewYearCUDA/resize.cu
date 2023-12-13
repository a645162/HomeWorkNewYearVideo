#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

// CUDA kernel function to resize the image using bilinear interpolation
__global__ void resizeImage(
        const uchar *src, uchar *dst,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        int channels
) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        auto scaleX = static_cast<float>(srcWidth) / dstWidth;
        auto scaleY = static_cast<float>(srcHeight) / dstHeight;

        auto srcX = x * scaleX;
        auto srcY = y * scaleY;

        auto x1 = static_cast<int>(srcX);
        auto y1 = static_cast<int>(srcY);
        auto x2 = x1 + 1;
        auto y2 = y1 + 1;

        auto xWeight = srcX - x1;
        auto yWeight = srcY - y1;

        for (int c = 0; c < channels; ++c) {
            float topLeft = src[(y1 * srcWidth + x1) * channels + c];
            float topRight = src[(y1 * srcWidth + x2) * channels + c];
            float bottomLeft = src[(y2 * srcWidth + x1) * channels + c];
            float bottomRight = src[(y2 * srcWidth + x2) * channels + c];

            float topInterpolation = topLeft * (1 - xWeight) + topRight * xWeight;
            float bottomInterpolation = bottomLeft * (1 - xWeight) + bottomRight * xWeight;

            dst[(y * dstWidth + x) * channels + c] = topInterpolation * (1 - yWeight) + bottomInterpolation * yWeight;
        }
    }
}

int main() {
    // Read input image
    cv::Mat image3 = cv::imread("shmtu_logo.png", cv::IMREAD_UNCHANGED);

    int srcWidth = image3.cols;
    int srcHeight = image3.rows;
    int channels = image3.channels();

    // Define the desired output size
    int dstWidth = 400;
    int dstHeight = 400;

    // Create and configure CUDA device memory
    uchar *devSrc, *devDst;
    cudaMalloc((void **) &devSrc, srcWidth * srcHeight * channels);
    cudaMalloc((void **) &devDst, dstWidth * dstHeight * channels);

    // Copy data from Mat to CUDA device memory
    cudaMemcpy(devSrc, image3.data, srcWidth * srcHeight * channels, cudaMemcpyHostToDevice);

    // Define thread block and grid size for CUDA kernel function
    dim3 blockSize(16, 16);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

    // Execute the CUDA kernel function
    resizeImage<<<gridSize, blockSize>>>(devSrc, devDst, srcWidth, srcHeight, dstWidth, dstHeight, channels);

    // Copy the result from CUDA device memory back to Mat
    cv::Mat result(dstHeight, dstWidth, CV_8UC(channels));
    cudaMemcpy(result.data, devDst, dstWidth * dstHeight * channels, cudaMemcpyDeviceToHost);

    // Free CUDA device memory
    cudaFree(devSrc);
    cudaFree(devDst);

    // Display the resized image
//    cv::imshow("Resized Image", result);
//    cv::waitKey(0);

    cv::imwrite("shmtu_logo_resized_400.png", result);

    return 0;
}