#include <iostream>
#include <fstream>
#include <CL/cl.h>
#include "../Devices/OpenCLDevices.h"

#define ROWS 50
#define COLS 50

int main() {
    // Initialize matrix
    int matrix[ROWS][COLS];

    int cpu_sum = 0;

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i][j] = i * COLS + j;
            cpu_sum += matrix[i][j];
        }
    }

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
    cl_mem bufferMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ROWS * COLS,
                                         matrix, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);

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

    // Execute the OpenCL kernel
    size_t global_size = ROWS * COLS;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    // Read the result back to the host
    int result;
    clEnqueueReadBuffer(command_queue, bufferResult, CL_TRUE, 0, sizeof(int), &result, 0, NULL, NULL);

    // Print the result
    std::cout << "CPU SUM: " << cpu_sum << std::endl;
    std::cout << "GPU SUM: " << result << std::endl;

    // Release resources
    clReleaseMemObject(bufferMatrix);
    clReleaseMemObject(bufferResult);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
