//
// Created by konghaomin on 23-12-13.
//

#include "ImageGaussianBlur.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>


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

    // Launch the 2D convolution kernel
//    convolution2D<<<gridSize, blockSize>>>(
//            d_input, d_output, height, width, channels, d_kernel, kernelSize, padSize
//    );

//    dim3 threadsPerBlock(channels, 1, 1);
//    dim3 blocksPerGrid((height - kernelSize + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                       (width - kernelSize + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
//
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width - kernelSize + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height - kernelSize + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (channels + threadsPerBlock.z - 1) / threadsPerBlock.z);
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_output, height, width, channels, d_kernel, kernelSize, padSize
    );

    cudaDeviceSynchronize();

    cudaMemcpy(output.ptr(), d_output, output.total() * output.elemSize(), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main() {
    // Read the input image using OpenCV
    cv::Mat originalImage = cv::imread("input.png");
    cv::resize(originalImage, originalImage, cv::Size(1000, 563));;

    if (originalImage.empty()) {
        std::cerr << "Error: Could not read the input image." << std::endl;
        return -1;
    }

    // Set the number of frames and frame rate
    int numFrames = 100;
    int frameRate = 10;  // frames per second

    // Create a VideoWriter object
    cv::VideoWriter videoWriter("blurred_video.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), frameRate,
                                cv::Size(originalImage.cols, originalImage.rows));

    // Generate frames with varying blur intensity
    for (int i = 70; i < numFrames; ++i) {
//    for (int i = 0; i < numFrames; ++i) {
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

//        if (i == 99) {
//            std::cout << "Frame index " << i << ": Blur Size = " << max_blur_size << ", Blur Strength = "
//                      << max_blur_strength << std::endl;
//        }

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

// main.cpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <CL/cl.h>
#include <chrono>
#include "kernels.cl"  // Include the OpenCL kernels

// ...（主程序的其余部分保持不变）

int main() {
    // ...（主程序的其余部分保持不变）

    // Load and compile OpenCL kernels
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Load and compile the OpenCL kernels
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Check for OpenCL program build errors

    cl_kernel kernel_matrixElementSum = clCreateKernel(program, "matrixElementSum", &err);
    cl_kernel kernel_matrixElementWiseDivision = clCreateKernel(program, "matrixElementWiseDivision", &err);
    cl_kernel kernel_generateGaussianKernel = clCreateKernel(program, "generateGaussianKernel", &err);
    cl_kernel kernel_convolution2D = clCreateKernel(program, "convolution2D", &err);

    // ...（主程序的其余部分保持不变）

    // Release OpenCL resources
    clReleaseKernel(kernel_matrixElementSum);
    clReleaseKernel(kernel_matrixElementWiseDivision);
    clReleaseKernel(kernel_generateGaussianKernel);
    clReleaseKernel(kernel_convolution2D);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
