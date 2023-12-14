//
// Created by konghaomin on 23-12-13.
//

#include "ImageConvolution.h"

#include <iostream>

#include "../../Author/Author.h"

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLFlow.h"
#include "../Include/OpenCLProgram.h"

#include "../Devices/OpenCLDevices.h"

#include "../Kernel/KernelConvolution.h"

#include <opencv2/opencv.hpp>

OpenCLProgram CLCreateProgramImageResize(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "convolution2D",
            cl_kernel_convolution
    };
}

void KernelSetArgImageConv(
        cl_kernel kernel,
        cl_mem devSrc,
        cl_mem devDst,
        int srcWidth,
        int srcHeight,
        int channels,
        cl_mem conv_kernel,
        int kernelSize,
        int padSize
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &srcWidth);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &srcHeight);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &conv_kernel);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &kernelSize);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &padSize);
}

void conv_demo(cl_context context, cl_device_id device) {

    // Read input image
    cv::Mat image3 = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
//    cv::Mat image3 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);
    cv::resize(image3, image3, cv::Size(image3.cols / 2, image3.rows / 2));

    // Convert to gray
    cv::cvtColor(image3, image3, cv::COLOR_BGR2GRAY);

    int srcWidth = image3.cols;
    int srcHeight = image3.rows;
    int channels = image3.channels();

    cl_command_queue queue = CLCreateCommandQueue(context, device);


//    cl_program program = CLCreateProgramImageResize(context, device);

    OpenCLProgram program_conv = CLCreateProgramImageResize(context, device);

    // Create OpenCL buffers for input and output data

    cl_mem devSrc = OpenCLMalloc(
            context,
            srcWidth * srcHeight * channels,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image3.data
    );

    cl_mem devDst = OpenCLMalloc(
            context,
            srcWidth * srcHeight * channels,
            CL_MEM_WRITE_ONLY,
            nullptr
    );

    const float kernel_laplacian[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    const int kernelSize = 3;
    const int padSize = kernelSize / 2;

    cl_mem devConvKernel = OpenCLMalloc(
            context,
            kernelSize * kernelSize * sizeof(float),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (void *) (kernel_laplacian)
    );

//    cl_kernel kernel = CLCreateKernelImageResize(program);
    cl_kernel kernel = program_conv.CreateKernel();

    KernelSetArgImageConv(
            kernel,
            devSrc, devDst,
            srcWidth, srcHeight, channels,
            devConvKernel, kernelSize, padSize
    );

    // Define global and local work sizes
    size_t globalWorkSize[2] = {
            static_cast<size_t>(srcWidth),
            static_cast<size_t>(srcHeight)
    };
    size_t localWorkSize[2] = {16, 16};

    // Execute the OpenCL kernel
    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize, localWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(srcHeight, srcWidth, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            srcWidth * srcHeight * channels
    );

    // Free OpenCL resources

    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDst);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Resized Image", result);
    cv::waitKey(0);
}

int main() {
//    KHM::sayHello();

    cl_device_id device = UserSelectDevice();

    auto max_work_group_size = CLGetInfoMaxWorkGroupSize(device);

    cl_context context =
            CLCreateContext(device);

    // resize demo
    conv_demo(context, device);

    clReleaseContext(context);

    return 0;
}
