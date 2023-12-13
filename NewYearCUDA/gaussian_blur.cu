#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846

// Gaussian kernel parameters
const int kernelSize = 5;

// Calculate Gaussian kernel
__host__ __device__ float gaussian(int x, int y, float sigma) {
    return exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlur(const uchar3* inputImage, uchar3* outputImage, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float r = 0.0f, g = 0.0f, b = 0.0f;

        // Apply the separable Gaussian filter
        for (int i = -kernelSize / 2; i <= kernelSize / 2; ++i) {
            for (int j = -kernelSize / 2; j <= kernelSize / 2; ++j) {
                int offsetX = x + i;
                int offsetY = y + j;

                // Ensure within bounds
                offsetX = max(0, min(width - 1, offsetX));
                offsetY = max(0, min(height - 1, offsetY));

                float weight = gaussian(i, j, sigma);

                r += weight * inputImage[offsetY * width + offsetX].x;
                g += weight * inputImage[offsetY * width + offsetX].y;
                b += weight * inputImage[offsetY * width + offsetX].z;
            }
        }

        // Save the result to the output image
        outputImage[y * width + x] = make_uchar3(static_cast<uchar>(r), static_cast<uchar>(g), static_cast<uchar>(b));
    }
}

int main() {
    // Read the image using OpenCV
    cv::Mat inputImage = cv::imread("gaussian.png");

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

    // Gaussian kernel parameter
    float sigma = 1.0;

    // Launch the CUDA kernel
    gaussianBlur<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height, sigma);

    // Copy the result back to the host
    cudaMemcpy(hostOutputImage, deviceOutputImage, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Create a new OpenCV image with the blurred data
    cv::Mat outputImage(height, width, CV_8UC3, hostOutputImage);

    // Save the blurred image
    cv::imwrite("blurred_image.jpg", outputImage);
    cv::imshow("Original Image", inputImage);
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    // Free memory
    delete[] hostOutputImage;
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    return 0;
}
