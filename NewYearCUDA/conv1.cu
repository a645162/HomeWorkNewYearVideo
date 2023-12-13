#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

__global__ void matrixElementSum(float *matrix, float *result, int size) {
    auto tid = threadIdx.x + threadIdx.y * blockDim.x;
    auto stride = blockDim.x * blockDim.y;

    // Perform block-wise reduction
    for (auto i = tid; i < size * size; i += stride) {
        atomicAdd(result, matrix[i]);
    }
}

__global__ void matrixElementWiseDivision(float *matrix, float *divisor, int size) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size) {
        matrix[y * size + x] /= *divisor;
    }
}

// Define CUDA kernel for Gaussian kernel generation
__global__ void generateGaussianKernel(float *kernel, int size, float strength) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size) {
        float center = static_cast<float>(size - 1) / 2;

        float value = static_cast<float>(
                              1.0f
                              /
                              (2.0f * M_PI * powf(strength, 2))
                      ) *
                      exp(
                              -(
                                      powf(static_cast<float>(x) - center, 2) +
                                      powf(static_cast<float>(y) - center, 2)
                              )
                              /
                              (2 * strength * strength)
                      );

        kernel[y * size + x] = value;  // Divide by strength to match Python implementation
    }
}

// Define CUDA kernel for 2D convolution
__global__ void convolution2D(const uchar *input, uchar *output, int height, int width, int channels,
                              const float *kernel, int kernelSize, int padSize) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - kernelSize + 1 && y < height - kernelSize + 1) {
        for (int c = 0; c < channels; ++c) {
            float current_channel_result = 0.0f;

            for (int i = 0; i < kernelSize; ++i) {
                for (int j = 0; j < kernelSize; ++j) {
                    int imageX = static_cast<int>(x) + i - padSize;
                    int imageY = static_cast<int>(y) + j - padSize;

                    if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                        current_channel_result += kernel[i * kernelSize + j] *
                                                  static_cast<float>(input[(imageY * width + imageX) * channels + c]);
                    }
                }
            }

            output[(y * width + x) * channels + c] = static_cast<uchar>(current_channel_result);
        }
    }
}

// Function to apply Gaussian blur using CUDA
void applyGaussianBlurCUDA(const cv::Mat &input, cv::Mat &output, int kernelSize, float strength) {

    auto width = input.cols;
    auto height = input.rows;

    int channels = input.channels();

    int padSize = kernelSize / 2;

    uchar *d_input, *d_output;
    float *d_kernel;

    cudaMalloc((void **) &d_input, input.total() * input.elemSize());
    cudaMalloc((void **) &d_output, output.total() * output.elemSize());
    cudaMalloc((void **) &d_kernel, kernelSize * kernelSize * sizeof(float));

    cudaMemcpy(d_input, input.ptr(), input.total() * input.elemSize(), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the Gaussian kernel generation kernel
    generateGaussianKernel<<<gridSize, blockSize>>>(d_kernel, kernelSize, strength);
    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // calc matrix sum
    float *dev_result;
    cudaMalloc((void **) &dev_result, sizeof(float));
    // Using 4x4 thread blocks for block-wise reduction
    dim3 atomAddBlockDim(4, 4);
    dim3 atomAddGridDim(1, 1);
    matrixElementSum<<<atomAddGridDim, atomAddBlockDim>>>(d_kernel, dev_result, kernelSize);
    cudaDeviceSynchronize();

    matrixElementWiseDivision<<<gridSize, blockSize>>>(d_kernel, dev_result, kernelSize);
    cudaDeviceSynchronize();
    cudaFree(dev_result);

//    if (kernelSize == 16 && strength == 5.1f) {
//        printf("kernel:\n");
//        float *kernel = new float[kernelSize * kernelSize];
//        cudaMemcpy(kernel, d_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyDeviceToHost);
//        for (int i = 0; i < kernelSize; ++i) {
//            for (int j = 0; j < kernelSize; ++j) {
//                printf("%f ", kernel[i * kernelSize + j]);
//            }
//            printf("\n");
//        }
//    }

    // Launch the 2D convolution kernel
    convolution2D<<<gridSize, blockSize>>>(d_input, d_output, rows, cols, channels, d_kernel, kernelSize, padSize);

    cudaMemcpy(output.ptr(), d_output, output.total() * output.elemSize(), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main() {
    // Read the input image using OpenCV
    cv::Mat originalImage = cv::imread("input.png");

    if (originalImage.empty()) {
        std::cerr << "Error: Could not read the input image." << std::endl;
        return -1;
    }

    // Set the number of frames and frame rate
    int numFrames = 100;
    int frameRate = 20;  // frames per second

    // Create a VideoWriter object
    cv::VideoWriter videoWriter("blurred_video.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), frameRate,
                                cv::Size(originalImage.cols, originalImage.rows));

    // Generate frames with varying blur intensity
//    for (int i = 50; i < 51; ++i) {
    for (int i = 0; i < numFrames; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Vary the blur intensity from 0 to 100
        float blur_intensity = static_cast<float>(i * 100) / (numFrames - 1);

        // Interpolate between [0, 100] to get the maximum blur size [1, 31]
        int max_blur_size = static_cast<int>(std::round((blur_intensity - 0) * (31 - 1) / (100 - 0) + 1));

        // Interpolate between [0, 100] to get the maximum blur strength [0.1, 10.0]
        float max_blur_strength = (blur_intensity - 0) * (10.0 - 0.1) / (100 - 0) + 0.1;

        // Output the values for the current frame

        // Apply Gaussian blur using CUDA
        cv::Mat frameBlurred(originalImage.size(), originalImage.type());
        applyGaussianBlurCUDA(originalImage, frameBlurred, max_blur_size, max_blur_strength);

        if (i == 99) {
            std::cout << "Frame " << i << ": Blur Size = " << max_blur_size << ", Blur Strength = "
                      << max_blur_strength << std::endl;

        }

        // Write the frame to the video
        videoWriter.write(frameBlurred);

        // Print the frame generation time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Frame " << i + 1 << "/" << numFrames << " - Time: " << duration.count() << " seconds"
                  << std::endl;
    }

    // Release the VideoWriter object
    videoWriter.release();

    std::cout << "Video created successfully." << std::endl;

    return 0;
}
