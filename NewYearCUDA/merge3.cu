#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>

__global__
void mergeImages(const uchar* image1, const uchar* image2, uchar* output, int width, int height, int mergeX, int mergeY, int image2Width, int image2Height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * 3;

        if (x >= mergeX && x < mergeX + image2Width && y >= mergeY && y < mergeY + image2Height)
        {
            int image2X = x - mergeX;
            int image2Y = y - mergeY;
            int image2Index = (image2Y * image2Width + image2X) * 3;

            output[index] = image2[image2Index];
            output[index + 1] = image2[image2Index + 1];
            output[index + 2] = image2[image2Index + 2];
        }
        else
        {
            output[index] = image1[index];
            output[index + 1] = image1[index + 1];
            output[index + 2] = image1[index + 2];
        }
    }
}

int main()
{
    cv::Mat image1 = cv::imread("image1.png");
    cv::Mat image2 = cv::imread("image2.png");

    int width = std::max(image1.cols, image2.cols);
    int height = std::max(image1.rows, image2.rows);

    cv::Mat outputImage(height, width, CV_8UC3);

    uchar* deviceImage1;
    uchar* deviceImage2;
    uchar* deviceOutputImage;

    cudaMalloc((void**)&deviceImage1, image1.total() * image1.elemSize());
    cudaMalloc((void**)&deviceImage2, image2.total() * image2.elemSize());
    cudaMalloc((void**)&deviceOutputImage, outputImage.total() * outputImage.elemSize());

    cudaMemcpy(deviceImage1, image1.data, image1.total() * image1.elemSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceImage2, image2.data, image2.total() * image2.elemSize(), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    int mergeX = 100;
    int mergeY = 100;
    int image2Width = image2.cols;
    int image2Height = image2.rows;
    mergeImages<<<gridSize, blockSize>>>(deviceImage1, deviceImage2, deviceOutputImage, width, height, mergeX, mergeY, image2Width, image2Height);

    cudaMemcpy(outputImage.data, deviceOutputImage, outputImage.total() * outputImage.elemSize(), cudaMemcpyDeviceToHost);

    cv::imshow("Image1", image1);
    cv::imshow("Image2", image2);
    cv::imshow("Merged Image", outputImage);
    cv::waitKey(0);

    cudaFree(deviceImage1);
    cudaFree(deviceImage2);
    cudaFree(deviceOutputImage);

    return 0;
}