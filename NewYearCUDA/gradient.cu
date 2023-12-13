#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

//#define CHECK_CUDA_ERROR(call) \
//    { \
//        cudaError_t cudaStatus = call; \
//        if (cudaStatus != cudaSuccess) { \
//            std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl; \
//            return cudaStatus; \
//        } \
//    }

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t cudaStatus = call; \
        if (cudaStatus != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__
void generateGradientColor(
        uchar *gradientColor, const int length,
        const int startR, const int startG, const int startB,
        const int endR, const int endG, const int endB,
        const unsigned int channels = 3,
        const uchar alpha = 255
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        float t = static_cast<float>(idx) / static_cast<float>(length - 1);

        auto color_index = channels * (idx);

        const auto colorR = static_cast<uchar>((1 - t) * static_cast<float>(startR) + t * static_cast<float>(endR));
        const auto colorG = static_cast<uchar>((1 - t) * static_cast<float>(startG) + t * static_cast<float>(endG));
        const auto colorB = static_cast<uchar>((1 - t) * static_cast<float>(startB) + t * static_cast<float>(endB));

        gradientColor[color_index + 0] = colorB;
        gradientColor[color_index + 1] = colorG;
        gradientColor[color_index + 2] = colorR;

        if (channels == 4) {
            gradientColor[color_index + 3] = alpha;
        }
    }
}

__global__
void generateGradientImage(
        uchar *gradientImage,
        const uchar *gradientColor, const int colorCount,
        const int width, const int height,
        const int center_x, const int center_y, const float maxR,
        const unsigned int channels = 3,
        const uchar alpha = 255
) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float r = sqrtf(
                powf(static_cast<float>(static_cast<int>(x) - center_x), 2)
                + powf(static_cast<float>(static_cast<int>(y) - center_y), 2)
        );
        const auto idx = static_cast<int>(r / maxR * static_cast<float>(colorCount - 1));

        const auto image_index = channels * (y * width + x);
        const auto color_index = channels * (idx);

        gradientImage[image_index + 0] = gradientColor[color_index + 0];
        gradientImage[image_index + 1] = gradientColor[color_index + 1];
        gradientImage[image_index + 2] = gradientColor[color_index + 2];

        if (channels == 4) {
            gradientImage[image_index + 3] = alpha;
        }
    }
}

uchar *generateGradientColor_GPU(
        const int colorCount,
        const int startR, const int startG, const int startB,
        const int endR, const int endG, const int endB,
        const unsigned int channels = 4,
        const uchar alpha = 255
) {
    uchar *d_gradientColor;

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_gradientColor, channels * colorCount * sizeof(uchar)));

    dim3 blockSize(256);
    dim3 gridSize((colorCount + blockSize.x - 1) / blockSize.x);

    generateGradientColor<<<gridSize, blockSize>>>(
            d_gradientColor, colorCount,
            startR, startG, startB,
            endR, endG, endB,
            channels
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return d_gradientColor;
}

uchar *generateGradientImage_GPU(
        const int width, const int height,
        uchar *gradientColor_GPU, const int colorCount,
        const int center_x, const int center_y,
        const unsigned int channels = 3,
        const uchar alpha = 255
) {

    uchar *d_gradientImage;

    auto maxR = std::max(
            {
                    sqrtf(
                            powf(static_cast<float>(center_x - 0), 2)
                            + powf(static_cast<float>(center_y - 0), 2)
                    ),
                    sqrtf(
                            powf(static_cast<float>(center_x - static_cast<int>(width)), 2)
                            + powf(static_cast<float>(center_y - 0), 2)
                    ),
                    sqrtf(
                            powf(static_cast<float>(center_x - 0), 2)
                            + powf(static_cast<float>(center_y - static_cast<int>(height)), 2)
                    ),
                    sqrtf(
                            powf(static_cast<float>(center_x - static_cast<int>(width)), 2)
                            + powf(static_cast<float>(center_y - static_cast<int>(height)), 2)
                    )
            }
    );

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_gradientImage, channels * width * height * sizeof(uchar)));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    generateGradientImage<<<grid_size, block_size>>>(
            d_gradientImage,
            gradientColor_GPU, colorCount,
            width, height,
            center_x, center_y, maxR,
            channels
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return d_gradientImage;
}

int main() {
    int width = 800;
    int height = 600;
    int colorCount = 256;
    int center_x = 200;
    int center_y = 100;

    int startR = 0, startG = 0, startB = 100;
    int endR = 0, endG = 200, endB = 200;

    const unsigned int channels = 4;

    uchar *d_gradientColor;
    uchar *d_gradientImage;

    d_gradientColor = generateGradientColor_GPU(
            colorCount,
            startR, startG, startB,
            endR, endG, endB,
            channels
    );

    d_gradientImage = generateGradientImage_GPU(
            width, height,
            d_gradientColor, colorCount,
            center_x, center_y,
            channels
    );

    cv::Mat gradientImage(height, width, CV_8UC(channels));
    CHECK_CUDA_ERROR(
            cudaMemcpy(
                    gradientImage.data, d_gradientImage,
                    channels * width * height * sizeof(uchar),
                    cudaMemcpyDeviceToHost
            )
    );

    cv::imshow("Color Gradient", gradientImage);
    cv::waitKey(0);

    cudaFree(d_gradientColor);
    cudaFree(d_gradientImage);

    return 0;
}
