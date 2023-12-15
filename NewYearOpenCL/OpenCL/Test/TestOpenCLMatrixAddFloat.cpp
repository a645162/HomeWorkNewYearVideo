// https://stackoverflow.com/questions/43028018/opencl-atomic-add-function-returns-wrong-value
// https://blog.csdn.net/pengx17/article/details/7876657

#include <iostream>
#include <fstream>
#include <CL/cl.h>
#include <chrono>

#include "../Devices/OpenCLDevices.h"

#define ROWS 31
#define COLS 31

#include <iostream>

//#include <eigen3/Eigen/Dense>

int main() {
    // Initialize matrix
    float matrix[ROWS][COLS];

//    // Map the array to an Eigen matrix
//    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
//            matrix1(matrix[0], ROWS, COLS);
//
//    // Calculate the sum of matrix elements
//
//    auto startCPU1 = std::chrono::high_resolution_clock::now();
//
//    float eigen_sum = matrix1.sum();
//
//    auto endCPU1 = std::chrono::high_resolution_clock::now();
//    auto durationCPU1 = std::chrono::duration_cast<std::chrono::microseconds>(endCPU1 - startCPU1).count();


    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i][j] = i * COLS + j;
        }
    }

    // Calculate CPU sum with timing
    auto startCPU = std::chrono::high_resolution_clock::now();

    float cpu_sum = 0;

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            cpu_sum += matrix[i][j];
        }
    }

    auto endCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU).count();

    // Read OpenCL kernel code from file
    std::ifstream kernelFile("matrix_add_kernel.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file!" << std::endl;
        return 1;
    }

    std::string kernelCode((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    kernelFile.close();

    auto device = UserSelectDevice();

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create OpenCL buffers for matrix and result
    auto time_gpu1 = std::chrono::high_resolution_clock::now();
    cl_mem bufferMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ROWS * COLS,
                                         matrix, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, NULL);
    auto time_gpu2 = std::chrono::high_resolution_clock::now();

    // Compile and create OpenCL kernel
    const char *kernelSource = kernelCode.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrixSum", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferResult);
    const int rows = ROWS, cols = COLS;
    clSetKernelArg(kernel, 2, sizeof(int), &rows);
    clSetKernelArg(kernel, 3, sizeof(int), &cols);

    // Execute the OpenCL kernel with timing
    auto startGPU = std::chrono::high_resolution_clock::now();

    size_t global_size = ROWS * COLS;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    auto endGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::microseconds>(endGPU - startGPU).count();

    // Read the result back to the host
    float result;
    clEnqueueReadBuffer(command_queue, bufferResult, CL_TRUE, 0, sizeof(float), &result, 0, NULL, NULL);

    // Print the result and timing information
    std::cout << "CPU For Sum: " << cpu_sum << " (Time: " << durationCPU << " microseconds)" << std::endl;
//    std::cout << "CPU eigen3 Sum: " << eigen_sum << " (Time: " << durationCPU1 << " microseconds)" << std::endl;
    std::cout << "GPU Sum: " << result << " (Time: " << durationGPU << " microseconds)" << std::endl;

    auto durationGPUCopy = std::chrono::duration_cast<std::chrono::microseconds>(time_gpu2 - time_gpu1).count();
    std::cout << "GPU Copy: " << durationGPUCopy << " microseconds" << std::endl;


    // Release resources
    clReleaseMemObject(bufferMatrix);
    clReleaseMemObject(bufferResult);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
