#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t cudaStatus = call; \
        if (cudaStatus != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__
void cropImage(
        const uchar *input, uchar *output,
        int inputWidth, int inputHeight,
        int outputWidth, int outputHeight,
        int x1, int y1, int x2, int y2,
        const int channels = 3
) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight) {
        auto inputX = x + x1;
        auto inputY = y + y1;

        auto inputIndex = (inputY * inputWidth + inputX) * 3; // Assuming 3 channels (RGB)
        auto outputIndex = (y * outputWidth + x) * 3;

        output[outputIndex + 0] = input[inputIndex + 0];
        output[outputIndex + 1] = input[inputIndex + 1];
        output[outputIndex + 2] = input[inputIndex + 2];
        if (channels == 4) {
            output[outputIndex + 3] = input[inputIndex + 3];
        }
    }
}

int main() {
    cv::Mat inputImage = cv::imread("image1.png");

    if (inputImage.empty()) {
        std::cerr << "Error: Unable to load input image." << std::endl;
        return EXIT_FAILURE;
    }

    int x1 = 100; // Define your cropping region
    int y1 = 50;
    int x2 = 600;
    int y2 = 500;

    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;

    int outputWidth = x2 - x1;
    int outputHeight = y2 - y1;

    cv::Mat outputImage(outputHeight, outputWidth, inputImage.type());

    uchar *d_inputImage;
    uchar *d_outputImage;

    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_inputImage, inputImage.total() * inputImage.elemSize()));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_outputImage, outputImage.total() * outputImage.elemSize()));

    CHECK_CUDA_ERROR(cudaMemcpy(d_inputImage, inputImage.data, inputImage.total() * inputImage.elemSize(),
                                cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

    cropImage<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, inputWidth, inputHeight, outputWidth, outputHeight,
                                       x1, y1, x2, y2);

    CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data, d_outputImage, outputImage.total() * outputImage.elemSize(),
                                cudaMemcpyDeviceToHost));

    cv::imshow("Input Image", inputImage);
    cv::imshow("Cropped Image", outputImage);
    cv::waitKey(0);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return EXIT_SUCCESS;
}
