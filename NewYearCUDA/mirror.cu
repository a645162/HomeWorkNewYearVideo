#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for horizontal mirroring
__global__ void robst(const uchar3* inputImage, uchar3* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Mirror horizontally
        outputImage[y * width + x] = inputImage[y * width + (width - 1 - x)];
    }
}

int main() {
    // Read the image using OpenCV
    cv::Mat inputImage = cv::imread("image1.png");

    if (inputImage.empty()) {
        std::cerr << "Failed to read the image." << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory on the host
    uchar3* hostInputImage = (uchar3*)inputImage.ptr();
    uchar3* hostOutputImage = new uchar3[width * height];

    // Allocate memory on the device
    uchar3* deviceInputImage;
    uchar3* deviceOutputImage;

    cudaMalloc((void**)&deviceInputImage, width * height * sizeof(uchar3));
    cudaMalloc((void**)&deviceOutputImage, width * height * sizeof(uchar3));

    // Copy the input image to the device
    cudaMemcpy(deviceInputImage, hostInputImage, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Specify block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    mirrorImage<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height);

    // Copy the result back to the host
    cudaMemcpy(hostOutputImage, deviceOutputImage, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Create a new OpenCV image with the mirrored data
    cv::Mat outputImage(height, width, CV_8UC3, hostOutputImage);

    // Display the original and mirrored images
    cv::imshow("Original Image", inputImage);
    cv::imshow("Mirrored Image", outputImage);
    cv::waitKey(0);

    // Free memory
    delete[] hostOutputImage;
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    return 0;
}
