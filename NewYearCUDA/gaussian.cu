#include <iostream>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

__global__ void generateRandomNumbers(float *data, int size, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, tid, 0, &state);
    data[tid] = curand_uniform(&state);
}

#include <cmath>

// CUDA kernel to add Gaussian noise to an image
__global__ void
addGaussianNoise(uchar3 *image, int width, int height, float *random_numbers, float mean, float stddev) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        curandState state;

        curand_init(123, x + y * width, 0, &state);
        float noise = mean + stddev * random_numbers[x + y * width];

        uchar3 pixel = image[y * width + x];
        pixel.x = static_cast<uchar>(min(255.0f, max(0.0f, pixel.x + noise)));

        curand_init(200, x + y * width, 0, &state);
        noise = mean + stddev * random_numbers[x + y * width];

        pixel.y = static_cast<uchar>(min(255.0f, max(0.0f, pixel.y + noise)));

        curand_init(300, x + y * width, 0, &state);
        noise = mean + stddev * random_numbers[x + y * width];

        pixel.z = static_cast<uchar>(min(255.0f, max(0.0f, pixel.z + noise)));

        image[y * width + x] = pixel;
    }
}


int main() {
    // Load an image using OpenCV
    cv::Mat input_image = cv::imread("input.png");
    cv::resize(input_image, input_image, cv::Size(1080, 607));

    if (input_image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    const int size = input_image.rows * input_image.cols;
    float *device_data;
    cudaMalloc((void **) &device_data, size * sizeof(float));

    // Set the seed for the random number generator
    unsigned long long seed = 123;

    // Call the CUDA kernel to generate random numbers
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    generateRandomNumbers<<<gridSize, blockSize>>>(device_data, size, seed);

    // Copy the image to the GPU
    uchar3 *gpu_image;
    cudaMalloc((void **) &gpu_image, size * sizeof(uchar3));
    cudaMemcpy(gpu_image, input_image.data, size * sizeof(uchar3), cudaMemcpyHostToDevice);

    // Call the CUDA kernel to add Gaussian noise to the image
    dim3 block_size(16, 16);
    dim3 grid_size((input_image.cols + block_size.x - 1) / block_size.x,
                   (input_image.rows + block_size.y - 1) / block_size.y);

    addGaussianNoise<<<grid_size, block_size>>>(gpu_image, input_image.cols, input_image.rows, device_data, 0.1f,
                                                25.0f);

    // Copy the image with noise back to the host
    cv::Mat output_image(input_image.rows, input_image.cols, CV_8UC3);
    cudaMemcpy(output_image.data, gpu_image, size * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cv::imwrite("gaussian.png", output_image);
    // Display the original and noisy images
    cv::imshow("Original Image", input_image);
    cv::imshow("Gaussian Image", output_image);
    cv::waitKey(0);

    // Free memory
    cudaFree(device_data);
    cudaFree(gpu_image);

    return 0;
}
