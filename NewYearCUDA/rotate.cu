#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define PI 3.14159265

__global__
void rotateImage(const uchar *input, uchar *output, int width, int height, float angle) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float radians = angle * PI / 180.0;
    float cosVal = cosf(radians);
    float sinVal = sinf(radians);

    int centerX = width / 2;
    int centerY = height / 2;

    int rotatedX = static_cast<int>(cosVal * (x - centerX) - sinVal * (y - centerY) + centerX);
    int rotatedY = static_cast<int>(sinVal * (x - centerX) + cosVal * (y - centerY) + centerY);

    if (rotatedX >= 0 && rotatedX < width && rotatedY >= 0 && rotatedY < height) {
        int inputIndex = (rotatedY * width + rotatedX) * 3;
        int outputIndex = (y * width + x) * 3;

        output[outputIndex] = input[inputIndex];
        output[outputIndex + 1] = input[inputIndex + 1];
        output[outputIndex + 2] = input[inputIndex + 2];
    }
}

int main() {
    int width = 400;
    int height = 400;
    int channels = 3;

    cv::Mat inputImage = cv::Mat::zeros(height, width, CV_8UC3);
    cv::putText(inputImage, "Haomin Kong", cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255),
                3);

    int imageSize = width * height * channels;
    uchar *deviceInputImage;
    uchar *deviceOutputImage;

    cudaMalloc((void **) &deviceInputImage, imageSize);
    cudaMalloc((void **) &deviceOutputImage, imageSize);
    cudaMemcpy(deviceInputImage, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    float angle = 30.0f;
    rotateImage<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height, angle);

    uchar *outputImage = new uchar[imageSize];
    cudaMemcpy(outputImage, deviceOutputImage, imageSize, cudaMemcpyDeviceToHost);

    cv::Mat rotatedImage(height, width, CV_8UC3, outputImage);
    cv::imshow("Original Image", inputImage);
    cv::imshow("Rotated Image", rotatedImage);
    cv::waitKey(0);

    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    delete[] outputImage;

    return 0;
}