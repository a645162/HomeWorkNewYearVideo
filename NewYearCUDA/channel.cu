#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to convert 3-channel image to 4-channel image
__global__ void convertTo4Channels(const uchar3* inputImage, uchar4* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar3 pixel = inputImage[y * width + x];

        // Set alpha channel to 255 for full opacity
        uchar4 resultPixel = make_uchar4(pixel.x, pixel.y, pixel.z, 255);

        outputImage[y * width + x] = resultPixel;
    }
}

int main() {
    // Read the 3-channel image using OpenCV
    cv::Mat inputImage = cv::imread("image1.png");

    if (inputImage.empty()) {
        std::cerr << "Failed to read the image." << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory on the host
    uchar3* hostInputImage = (uchar3*)inputImage.ptr();
    uchar4* hostOutputImage = new uchar4[width * height];

    // Allocate memory on the device
    uchar3* deviceInputImage;
    uchar4* deviceOutputImage;

    cudaMalloc((void**)&deviceInputImage, width * height * sizeof(uchar3));
    cudaMalloc((void**)&deviceOutputImage, width * height * sizeof(uchar4));

    // Copy the input image to the device
    cudaMemcpy(deviceInputImage, hostInputImage, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Specify block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    convertTo4Channels<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height);

    // Copy the result back to the host
    cudaMemcpy(hostOutputImage, deviceOutputImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // Create a new OpenCV image with the 4-channel data
    cv::Mat outputImage(height, width, CV_8UC4, hostOutputImage);

    // Display the original and converted images
    cv::imshow("Original Image", inputImage);
    cv::imshow("4-Channel Image", outputImage);
    cv::waitKey(0);

    // Free memory
    delete[] hostOutputImage;
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    return 0;
}
