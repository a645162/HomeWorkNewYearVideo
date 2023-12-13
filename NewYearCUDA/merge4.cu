#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

// Merge Image, image1 is the background image, image2 is the foreground image
__global__
void mergeImages(
        const uchar *image1, const uchar *image2, uchar *output,
        const int image1_width, const int image1_height, const int image1_channels,
        const int image2_target_x, const int image2_target_y,
        const int image2_width, const int image2_height, const int image2_channels,
        const int image2_alpha = 255
) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < image1_width && y < image1_height) {
        const auto index = (y * image1_width + x) * image1_channels;

        if (
                (
                        static_cast<int>(x) >= image2_target_x &&
                        x < static_cast<unsigned int>(image2_target_x + static_cast<int>(image2_width))
                )
                &&
                (
                        static_cast<int>(y) >= image2_target_y &&
                        y < static_cast<unsigned int>(image2_target_y + static_cast<int>(image2_height))
                )
                ) {
            auto image2_X = static_cast<unsigned int>(static_cast<int>(x) - image2_target_x);
            auto image2_Y = static_cast<unsigned int>(static_cast<int>(y) - image2_target_y);

            // Using 4 channels for BGRA
            auto image2_index = (image2_Y * image2_width + image2_X) * image2_channels;

            if (image2_channels == 4) {
                // Normalize alpha to range [0, 1]
                const float alpha = static_cast<float>(image2[image2_index + 3]) / 255.0f;
                const float user_set_alpha = static_cast<float>(image2_alpha) / 255.0f;

                output[index] = static_cast<uchar>(
                        static_cast<float>(image1[index]) * (1.0f - alpha)
                        +
                        static_cast<float>(image2[image2_index]) * alpha * user_set_alpha
                );
                output[index + 1] = static_cast<uchar>(
                        static_cast<float>(image1[index + 1]) * (1.0f - alpha)
                        +
                        static_cast<float>(image2[image2_index + 1]) * alpha * user_set_alpha
                );
                output[index + 2] = static_cast<uchar>(
                        static_cast<float>(image1[index + 2]) * (1.0f - alpha)
                        +
                        static_cast<float>(image2[image2_index + 2]) * alpha * user_set_alpha
                );
            } else {
                output[index] = static_cast<uchar>(image2[image2_index]);
                output[index + 1] = static_cast<uchar>(image2[image2_index + 1]);
                output[index + 2] = static_cast<uchar>(image2[image2_index + 2]);
            }
        } else {
            output[index] = image1[index];
            output[index + 1] = image1[index + 1];
            output[index + 2] = image1[index + 2];
        }

        // Copy image1 alpha channel to output if it exists
        if (image1_channels == 4) {
            output[index + 3] = static_cast<uchar>(image1[index + 3]);
        }
    }
}

int main() {
    cv::Mat image1 = cv::imread("image1.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("image2.png", cv::IMREAD_UNCHANGED);
//    cv::Mat image1 = cv::imread("image1.png");
//    cv::Mat image2 = cv::imread("image2.png");

//    int width = std::max(image1.cols, image2.cols);
//    int height = std::max(image1.rows, image2.rows);

    const auto width = image1.cols;
    const auto height = image1.rows;

    std::cout << "width " << width << std::endl;
    std::cout << "height " << height << std::endl;

    std::cout << "image1.total() " << image1.total() << std::endl;
    std::cout << "image1.elemSize() " << image1.elemSize() << std::endl;
    std::cout << "image1.channels() " << image1.channels() << std::endl;

    std::cout << "image2.total() " << image2.total() << std::endl;
    std::cout << "image2.elemSize() " << image2.elemSize() << std::endl;
    std::cout << "image2.channels() " << image2.channels() << std::endl;

    const auto channels = image1.channels();
    cv::Mat outputImage(height, width, CV_8UC(channels));

    uchar *deviceImage1;
    uchar *deviceImage2;
    uchar *deviceOutputImage;

    cudaMalloc((void **) &deviceImage1, image1.total() * image1.elemSize());
    cudaMalloc((void **) &deviceImage2, image2.total() * image2.elemSize());
    cudaMalloc((void **) &deviceOutputImage, outputImage.total() * outputImage.elemSize());

    cudaMemcpy(
            deviceImage1,
            image1.data,
            image1.total() * image1.elemSize(),
            cudaMemcpyHostToDevice
    );
    cudaMemcpy(
            deviceImage2,
            image2.data,
            image2.total() * image2.elemSize(),
            cudaMemcpyHostToDevice
    );

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

//    int mergeX = 100;
//    int mergeY = 100;
    int mergeX = -100;
    int mergeY = -100;

    int image2Width = image2.cols;
    int image2Height = image2.rows;


    std::cout << "=====Kernel Func Start=====" << std::endl;

    mergeImages<<<gridSize, blockSize>>>(
            deviceImage1, deviceImage2, deviceOutputImage,
            width, height, image1.channels(),
            mergeX, mergeY,
            image2Width, image2Height, image2.channels(),
            100
    );

    std::cout << "=====Kernel Func end!=====" << std::endl;

    cudaMemcpy(
            outputImage.data,
            deviceOutputImage,
            outputImage.total() * outputImage.elemSize(),
            cudaMemcpyDeviceToHost
    );

    std::cout << "=====Copy back to CPU end!=====" << std::endl;

    cv::imshow("Image1", image1);
    cv::imshow("Image2", image2);
    cv::imshow("Merged Image", outputImage);
    cv::waitKey(0);

    cudaFree(deviceImage1);
    cudaFree(deviceImage2);
    cudaFree(deviceOutputImage);

    return 0;
}
