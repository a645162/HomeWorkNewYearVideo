#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to convert 3-channel image to 4-channel image
__global__ void convert3To4Channels(const uchar3 *inputImage, uchar4 *outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar3 pixel = inputImage[y * width + x];

        // Set alpha channel to 255 for full opacity
        uchar4 resultPixel = make_uchar4(pixel.x, pixel.y, pixel.z, 255);

        outputImage[y * width + x] = resultPixel;
    }
}

__global__ void convert4To3Channels(const uchar4 *inputImage, uchar3 *outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 pixel = inputImage[y * width + x];
        outputImage[y * width + x] = make_uchar3(pixel.x, pixel.y, pixel.z);
    }
}

int main() {
    // Read the 3-channel image using OpenCV
    cv::Mat inputImage = cv::imread("../Resources/input.png");
    cv::resize(
            inputImage, inputImage,
            cv::Size(inputImage.cols / 4, inputImage.rows / 4)
    );

    if (inputImage.empty()) {
        std::cerr << "Failed to read the image." << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory on the host
    uchar3 *hostInputImage = (uchar3 *) inputImage.ptr();

    uchar4 *hostOutputImage = new uchar4[width * height];
    uchar4 *hostOutputImage3 = new uchar4[width * height];

    // Allocate memory on the device
    uchar3 *deviceInputImage;
    uchar4 *deviceOutputImage;

    uchar3 *deviceOutput3Image;

    cudaMalloc((void **) &deviceInputImage, width * height * sizeof(uchar3));
    cudaMalloc((void **) &deviceOutputImage, width * height * sizeof(uchar4));

    cudaMalloc((void **) &deviceOutput3Image, width * height * sizeof(uchar3));

    // Copy the input image to the device
    cudaMemcpy(deviceInputImage, hostInputImage, width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Specify block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    convert3To4Channels<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height);

    convert4To3Channels<<<gridSize, blockSize>>>(deviceOutputImage, deviceOutput3Image, width, height);

    // Copy the result back to the host
    cudaMemcpy(hostOutputImage, deviceOutputImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // Create a new OpenCV image with the 4-channel data
    cv::Mat outputImage(height, width, CV_8UC4, hostOutputImage);

    cudaMemcpy(hostOutputImage3, deviceOutput3Image, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);
    cv::Mat outputImage3(height, width, CV_8UC3, hostOutputImage3);

    // Display the original and converted images
    std::cout << "Original Image" << std::endl;
    std::cout << inputImage.channels() << std::endl;
    cv::imshow("Original Image", inputImage);


    std::cout << "4Channel Image" << std::endl;
    std::cout << outputImage.channels() << std::endl;
    cv::imshow("4Channel Image", outputImage);

    std::cout << "3Channel Image" << std::endl;
    std::cout << outputImage3.channels() << std::endl;
    cv::imshow("3Channel Image", outputImage3);

    cv::waitKey(0);

    // Free memory
    delete[] hostOutputImage;
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    return 0;
}
