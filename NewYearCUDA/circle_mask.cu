#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__
void MaskImageCircle(
        const uchar *input, uchar *output,
        const int width, const int height, const int channels,
        const int centerX, const int centerY, const float radius,
        bool focus_lamp = false,
        const int light_source_x = 0, const int light_source_y = 0,
        const float k_1 = 0, const float k_2 = 0, const float max_distance = 0,
        uchar3 focus_color = {0, 0, 0},
        uchar color_alpha = 150
) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const auto index = (y * width + x) * channels;

        const auto distance =
                sqrtf(
                        powf(static_cast<float>(static_cast<int>(x) - centerX), 2)
                        +
                        powf(static_cast<float>(static_cast<int>(y) - centerY), 2)
                );

        if (distance > static_cast<float>(radius)) {
            output[index + 3] = 0;

            if (focus_lamp) {
                const auto x_1 =
                        (1 / k_1) * static_cast<float>(y - light_source_y) + static_cast<float>(light_source_x);
                const auto x_2 =
                        (1 / k_2) * static_cast<float>(y - light_source_y) + static_cast<float>(light_source_x);

                const auto x_left = fminf(x_1, x_2);
                const auto x_right = fmaxf(x_1, x_2);

                const auto distance_current = sqrtf(
                        powf(static_cast<float>(static_cast<int>(x) - light_source_x), 2)
                        +
                        powf(static_cast<float>(static_cast<int>(y) - light_source_y), 2)
                );

                if (
                        x_left <= static_cast<float>(x) && static_cast<float>(x) <= x_right
                        &&
                        distance_current <= max_distance
                        ) {
                    const auto color_alpha_rate = static_cast<float>(color_alpha) / 255.0f;

                    output[index + 0] = static_cast<uchar>(
                            static_cast<float>(input[index + 0]) * (1 - color_alpha_rate) +
                            static_cast<float>(focus_color.x) * color_alpha_rate
                    );
                    output[index + 1] = static_cast<uchar>(
                            static_cast<float>(input[index + 1]) * (1 - color_alpha_rate) +
                            static_cast<float>(focus_color.y) * color_alpha_rate
                    );
                    output[index + 2] = static_cast<uchar>(
                            static_cast<float>(input[index + 2]) * (1 - color_alpha_rate) +
                            static_cast<float>(focus_color.z) * color_alpha_rate
                    );
                    if (channels == 4) {
                        output[index + 3] = color_alpha;
                    }
                }
            }

        } else {
            output[index] = input[index];
            output[index + 1] = input[index + 1];
            output[index + 2] = input[index + 2];
            output[index + 3] = input[index + 3];
        }
    }
}

void processImageCuda(
        unsigned char *h_input, unsigned char *h_output,
        int width, int height, int channels,
        float radius
) {
    auto size = width * height * channels * sizeof(unsigned char);
    unsigned char *d_input, *d_output;

    cudaMalloc((void **) &d_input, size);
    cudaMalloc((void **) &d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    const auto centerX = 2 * width / 3;
    const auto centerY = height / 2;

    const auto light_source_x = (width / 2);
    const auto light_source_y = -100;

    const auto k_center =
            (static_cast<float>(centerY - light_source_y)) / (static_cast<float>(centerX - light_source_x));
    const auto angle_center = atanf(k_center);
    const auto distance_center = sqrtf(
            powf(static_cast<float>(light_source_x - centerX), 2) +
            powf(static_cast<float>(light_source_y - centerY), 2)
    );
    const auto max_distance = sqrtf(
            powf(distance_center, 2) - powf(radius, 2)
    );

    const auto angle_between_center = asinf(radius / distance_center);
    const auto angle_1 = angle_center - angle_between_center;
    const auto angle_2 = angle_center + angle_between_center;

    const auto k_1 = tanf(angle_1);
    const auto k_2 = tanf(angle_2);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    MaskImageCircle<<<gridSize, blockSize>>>(
            d_input, d_output,
            width, height, channels,
            centerX, centerY,
            radius,
            true,
            light_source_x, light_source_y,
            k_1, k_2, max_distance,
            {0, 0, 0}
    );

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat image = cv::imread("input.png", cv::IMREAD_UNCHANGED);
    cv::resize(image, image, cv::Size(1080, 607));

    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    const float radius = 150.5f;  // Set your desired radius

    uchar *h_input = image.data;

    // Save h_output as your result image
    cv::Mat result(height, width, CV_8UC4);

    processImageCuda(h_input, result.data, width, height, channels, radius);
//    cv::imwrite("output_image.png", result);

    cv::imshow("Input", image);
    cv::imshow("Output", result);
    cv::waitKey(0);

    return 0;
}
