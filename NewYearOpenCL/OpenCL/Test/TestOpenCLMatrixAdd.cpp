//
// Created by konghaomin on 23-12-15.
//

#include <iostream>
#include <CL/cl.h>
#include <chrono>

#define ROWS 3
#define COLS 3

// OpenCL kernel to compute the sum of matrix elements
const char* kernelSource =
        "__kernel void matrixSum(__global float* matrix, __global float* result, const int rows, const int cols) {\n"
        "    int global_id = get_global_id(0);\n"
        "    int row = global_id / cols;\n"
        "    int col = global_id % cols;\n"
        "    if (row < rows && col < cols) {\n"
        "        atomic_add(&result[0], matrix[row * cols + col]);\n"
        "    }\n"
        "}\n";

int main1() {
    // Initialize matrix
    float matrix[ROWS][COLS] = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
    };

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create OpenCL buffers for matrix and result
    cl_mem matrixBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ROWS * COLS, matrix, NULL);
    cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

    // Compile and create OpenCL kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrixSum", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &ROWS);
    clSetKernelArg(kernel, 3, sizeof(int), &COLS);

    // Execute the OpenCL kernel
    size_t global_size = ROWS * COLS;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    // Read the result back to the host
    float resultGPU;
    clEnqueueReadBuffer(command_queue, resultBuffer, CL_TRUE, 0, sizeof(float), &resultGPU, 0, NULL, NULL);

    // CPU computation for comparison
    auto startCPU = std::chrono::high_resolution_clock::now();
    float resultCPU = 0.0f;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            resultCPU += matrix[i][j];
        }
    }
    auto endCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU).count();

    // Print results
    std::cout << "GPU Result: " << resultGPU << std::endl;
    std::cout << "CPU Result: " << resultCPU << std::endl;
    std::cout << "GPU Time: " << durationGPU << " microseconds" << std::endl;
    std::cout << "CPU Time: " << durationCPU << " microseconds" << std::endl;

    // Release resources
    clReleaseMemObject(matrixBuffer);
    clReleaseMemObject(resultBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
