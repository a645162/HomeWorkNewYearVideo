#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// CUDA kernel function: Compute the color values for each pixel in the image with sine waves along rectangle edges
__global__ void drawRectangle(
        uchar *d_image,
        const int width, const int height,
        const int x_1, const int y_1,
        const int x_2, const int y_2,
        const int thickness,
        uchar3 board_color,
        uchar3 fill_color,
        const unsigned int channels = 3,
        const bool fill = true,
        const bool sine_waves_board = false,
        const float frequency = 0.02f
) {

    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    const auto x1 = min(x_1, x_2);
    const auto y1 = min(y_1, y_2);

    const auto x2 = max(x_1, x_2);
    const auto y2 = max(y_1, y_2);

    auto index = (y * width + x) * channels;


    if (
            (x1 <= static_cast<int>(x) && static_cast<int>(x) <= x2)
            &&
            (y1 <= static_cast<int>(y) && static_cast<int>(y) <= y2)
            ) {
        // Draw Area

        // Fill the rectangle
        if (thickness <= 0) {
            if (fill) {
                d_image[index + 0] = fill_color.x;
                d_image[index + 1] = fill_color.y;
                d_image[index + 2] = fill_color.z;
            }
            return;
        }

        const auto fill_x1 = x1 + thickness;
        const auto fill_y1 = y1 + thickness;

        const auto fill_x2 = x2 - thickness;
        const auto fill_y2 = y2 - thickness;

        if (
                (fill_x1 <= static_cast<int>(x) && static_cast<int>(x) <= fill_x2)
                &&
                (fill_y1 <= static_cast<int>(y) && static_cast<int>(y) <= fill_y2)
                ) {
            // Fill Area
            if (fill) {
                d_image[index + 0] = fill_color.x;
                d_image[index + 1] = fill_color.y;
                d_image[index + 2] = fill_color.z;
            }
        } else {
            // Bord Area
            uchar3 current_board_color = board_color;

            if (sine_waves_board) {
                // Draw sine waves along the four edges of the rectangle
                float sineValue;

                // X Fixed Vertical Line
                if (
                        (x1 <= static_cast<int>(x) && static_cast<int>(x) <= x1 + thickness)
                        ||
                        (x2 - thickness <= static_cast<int> (x) && static_cast<int>(x) <= x2)
                        ) {
                    sineValue = sinf(static_cast<float>(y) * frequency);
                }

                // Y Fixed Horizontal Line
                if (
                        (y1 <= static_cast<int> (y) && static_cast<int>(y) <= y1 + thickness)
                        ||
                        (y2 - thickness <= static_cast<int>(y) && static_cast<int>(y) <= y2)
                        ) {
                    sineValue = sinf(static_cast<float>(x) * frequency);
                }

                // Normalize the sine value to the range [0, 1]
                sineValue += 1;
                sineValue /= 2;

                const auto color_x =
                        static_cast<int>(sineValue * static_cast<float>(current_board_color.x));
                const auto color_y =
                        static_cast<int>(sineValue * static_cast<float>(current_board_color.y));
                const auto color_z =
                        static_cast<int>(sineValue * static_cast<float>(current_board_color.z));

                current_board_color = make_uchar3(color_x, color_y, color_z);
            }

            d_image[index + 0] = current_board_color.x;
            d_image[index + 1] = current_board_color.y;
            d_image[index + 2] = current_board_color.z;
        }

    }
}

int main() {
    // Image dimensions
    int width = 800;
    int height = 600;
    int channels = 3;

    cv::Mat image1 = cv::imread("../Resources/input.png", cv::IMREAD_UNCHANGED);
    cv::cvtColor(image1, image1, cv::COLOR_BGRA2BGR);
    cv::resize(image1, image1, cv::Size(image1.cols / 4, image1.rows / 4));
    width = image1.cols;
    height = image1.rows;
    channels = image1.channels();

    // Allocate image memory on the host
//    uchar *h_image = new uchar[width * height * channels];

    // Allocate image memory on the CUDA device
    uchar *d_image;
    cudaMalloc((void **) &d_image, sizeof(uchar) * width * height * channels);

    cudaMemcpy(
            d_image, image1.data,
            sizeof(uchar) * width * height * channels,
            cudaMemcpyHostToDevice
    );

    // Set the frequency of the sine waves
    float frequency = 0.02;

    // Set CUDA kernel launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Execute the CUDA kernel function
    drawRectangle<<<gridSize, blockSize>>>(
            d_image,
            width, height,
            100, 100,
            700, 500,
            10,
            make_uchar3(255, 255, 255),
            make_uchar3(0, 0, 255),
            channels,
            true,
            true,
            frequency
    );

    cv::Mat image(height, width, CV_8UC(channels));

    // Transfer image data back to the host
    cudaMemcpy(image.data, d_image, sizeof(uchar) * width * height * channels, cudaMemcpyDeviceToHost);

    // Free image memory on the CUDA device
    cudaFree(d_image);

    // Display the image using OpenCV
    cv::imshow("Sine Waves on Rectangle Edges", image);
    cv::waitKey(0);

    // Free image memory on the host
//    delete[] h_image;

    return 0;
}
