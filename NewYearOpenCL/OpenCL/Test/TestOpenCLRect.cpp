//
// Created by konghaomin on 23-12-12.
//

#include <iostream>
#include <CL/cl.h>

#include <opencv2/opencv.hpp>


cl_device_id getOpenCLDevice() {
    // Get available OpenCL platforms
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Choose the third platform (index 2)
    cl_platform_id platform = platforms[2];

    // Get available devices on the platform
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

    // Choose the first device
    cl_device_id device = devices[0];

    // Output device name
    size_t deviceNameSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
    std::vector<char> deviceName(deviceNameSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);
    std::cout << "Using OpenCL device: " << deviceName.data() << std::endl;

    return device;
}


void drawRectangle(cl_device_id device, cl_context context, cl_command_queue queue, cv::Mat &image) {
    // Create buffer for image data
    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(uchar) * image.rows * image.cols * image.channels(), image.data,
                                        nullptr);

    // Define rectangle parameters
    int rectX = 100;
    int rectY = 100;
    int rectWidth = 300;
    int rectHeight = 300;

    // Create OpenCL program source code
    const char *sourceCode = R"(
        __kernel void drawRectangle(__global uchar* image, const int width, const int height,
                                    const int rectX, const int rectY, const int rectWidth, const int rectHeight) {
            int gidX = get_global_id(0);
            int gidY = get_global_id(1);

            if (gidX >= rectX && gidX < rectX + rectWidth && gidY >= rectY && gidY < rectY + rectHeight) {
                int index = (gidY * width + gidX) * 3; // Assuming 3 channels (BGR)
                image[index] = 0; // Blue
                image[index + 1] = 255; // Green
                image[index + 2] = 255; // Red
            }
        }
    )";

    // Create OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &sourceCode, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "drawRectangle", nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    clSetKernelArg(kernel, 1, sizeof(int), &image.cols);
    clSetKernelArg(kernel, 2, sizeof(int), &image.rows);
    clSetKernelArg(kernel, 3, sizeof(int), &rectX);
    clSetKernelArg(kernel, 4, sizeof(int), &rectY);
    clSetKernelArg(kernel, 5, sizeof(int), &rectWidth);
    clSetKernelArg(kernel, 6, sizeof(int), &rectHeight);

    // Execute kernel
    size_t globalSize[2] = {static_cast<size_t>(image.cols), static_cast<size_t>(image.rows)};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    // Read back the modified image
    clEnqueueReadBuffer(queue, imageBuffer, CL_TRUE, 0,
                        sizeof(uchar) * image.rows * image.cols * image.channels(), image.data, 0, nullptr, nullptr);

    // Clean up
    clReleaseMemObject(imageBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

int main() {
    // Get OpenCL device
    cl_device_id device = getOpenCLDevice();

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Create an empty image (500x500, 3 channels)
    cv::Mat image(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw a rectangle using OpenCL
    drawRectangle(device, context, queue, image);

    // Display the image using OpenCV
    cv::imshow("OpenCL Rectangle", image);
    cv::waitKey(0);

    // Clean up
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
