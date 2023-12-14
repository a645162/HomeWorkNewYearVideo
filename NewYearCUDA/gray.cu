#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define CONVERT_TO_GRAY_AVG(r, g, b) (uchar)((r + g + b) / 3)

#define WEIGHTED_GRAY_WEIGHT_R 0.299
#define WEIGHTED_GRAY_WEIGHT_G 0.587
#define WEIGHTED_GRAY_WEIGHT_B 0.114

#define CONVERT_TO_WEIGHTED_GRAY(r, g, b) (uchar)( \
    WEIGHTED_GRAY_WEIGHT_R * (r) + WEIGHTED_GRAY_WEIGHT_G * (g) + WEIGHTED_GRAY_WEIGHT_B * (b) \
)


// 0: average 1: weighted
__global__ void convertToGrayRGB(
        const uchar *inputImage, uchar *outputImage,
        int width, int height, int channels,
        int type = 0
) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        auto index = y * width + x;

        // Calculate pixel index for each channel
        auto blueIndex = index * channels;
        auto greenIndex = blueIndex + 1;
        auto redIndex = blueIndex + 2;
        auto alphaIndex = blueIndex + 3;

        auto r = inputImage[redIndex];
        auto g = inputImage[greenIndex];
        auto b = inputImage[blueIndex];

        uchar grayValue;

        if (type == 0) {
            grayValue = CONVERT_TO_GRAY_AVG(r, g, b);
        } else {
            grayValue = CONVERT_TO_WEIGHTED_GRAY(r, g, b);
        }

        if (channels == 4) {
            outputImage[alphaIndex] = inputImage[alphaIndex];
        } else if (channels != 3) {
            grayValue = 0;
        }

        // Set the same grayscale value for all channels
        outputImage[redIndex] = grayValue;
        outputImage[greenIndex] = grayValue;
        outputImage[blueIndex] = grayValue;
    }
}

__global__ void convertToGraySingleChannel(
        const uchar *inputImage, uchar *outputImage,
        int width, int height, int channels,
        int type = 0
) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        auto targetIndex = y * width + x;

        // Calculate pixel index for each channel
        auto blueIndex = targetIndex * channels;
        auto greenIndex = blueIndex + 1;
        auto redIndex = blueIndex + 2;

        auto r = inputImage[redIndex];
        auto g = inputImage[greenIndex];
        auto b = inputImage[blueIndex];

        uchar grayValue;

        if (type == 0) {
            grayValue = CONVERT_TO_GRAY_AVG(r, g, b);
        } else {
            grayValue = CONVERT_TO_WEIGHTED_GRAY(r, g, b);
        }

        outputImage[targetIndex] = grayValue;
    }
}

int main() {
    // Read the image using OpenCV
    cv::Mat inputImage = cv::imread("../Resources/input.png", cv::IMREAD_UNCHANGED);
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
    int channels = inputImage.channels(); // Get the number of channels

    std::cout << "channels: " << channels << std::endl;

    // Allocate memory on the host
    uchar *hostInputImage = inputImage.data;
    uchar *hostOutputImage = new uchar[inputImage.total() * channels];
    uchar *hostOutputImage1 = new uchar[inputImage.total() * channels];

    // Allocate memory on the device
    uchar *deviceInputImage;
    uchar *deviceOutputImage;
    uchar *deviceOutput1ChannelImage;

    cudaMalloc((void **) &deviceInputImage, inputImage.total() * channels * sizeof(uchar));
    cudaMalloc((void **) &deviceOutputImage, inputImage.total() * channels * sizeof(uchar));

    cudaMalloc((void **) &deviceOutput1ChannelImage, inputImage.total() * sizeof(uchar));

    // Copy the input image to the device
    cudaMemcpy(deviceInputImage, hostInputImage, inputImage.total() * channels * sizeof(uchar), cudaMemcpyHostToDevice);

    // Specify block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
//    convertToGrayRGB<<<gridSize, blockSize>>>(
//            deviceInputImage, deviceOutputImage,
//            width, height, channels,
//            1
//    );

    convertToGrayRGB<<<gridSize, blockSize>>>(
            deviceInputImage, deviceOutputImage,
            width, height, channels,
            1
    );

    convertToGraySingleChannel<<<gridSize, blockSize>>>(
            deviceInputImage, deviceOutput1ChannelImage,
            width, height, channels,
            1
    );

    // Copy the result back to the host
    cudaMemcpy(hostOutputImage, deviceOutputImage, inputImage.total() * channels * sizeof(uchar),
               cudaMemcpyDeviceToHost);
//
//    // Create a new OpenCV image with the grayscale data
    cv::Mat outputImage(height, width, CV_8UC(channels), hostOutputImage);

    cudaMemcpy(hostOutputImage1, deviceOutput1ChannelImage, inputImage.total() * sizeof(uchar),
               cudaMemcpyDeviceToHost);
    cv::Mat outputImage1(height, width, CV_8UC1, hostOutputImage1);

    // Display the original and grayscale images
    cv::imshow("Original Image", inputImage);
    cv::imshow("Grayscale Image", outputImage);
    cv::imshow("Grayscale Image1", outputImage1);
    cv::waitKey(0);

    // Free memory
    delete[] hostOutputImage;
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
}