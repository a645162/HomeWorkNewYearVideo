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

OpenCLProgram CLCreateProgramImageConv(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "convolution2Dim",
            cl_kernel_convolution
    };
}

void KernelSetArgImageConv(
        cl_kernel kernel,
        cl_mem device_src,
        cl_mem device_dst,
        int width,
        int height,
        int channels,
        cl_mem conv_kernel,
        int conv_kernel_size,
        int padSize
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_src);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_dst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &conv_kernel);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &conv_kernel_size);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &padSize);
}

void conv_demo(cl_context context, cl_device_id device) {

    // Read input image
    cv::Mat image3 = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
//    cv::Mat image3 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);
    cv::resize(image3, image3, cv::Size(image3.cols / 4, image3.rows / 4));

    // Convert to gray
    cv::cvtColor(image3, image3, cv::COLOR_BGR2GRAY);

    int width = image3.cols;
    int height = image3.rows;
    int channels = image3.channels();

    std::cout << "Image size: " << width << "x" << height << "x" << channels << std::endl;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

//    cl_program program = CLCreateProgramImageResize(context, device);

    OpenCLProgram program_conv = CLCreateProgramImageConv(context, device);

    // Create OpenCL buffers for input and output data

    const auto data_size = width * height * channels * sizeof(uchar);

    cl_mem devSrc = OpenCLMalloc(
            context,
            data_size,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image3.data
    );

    cl_mem devDst = OpenCLMalloc(
            context,
            data_size,
            CL_MEM_WRITE_ONLY,
            nullptr
    );

//    const float kernel_laplacian[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    const float kernel_laplacian[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
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
            width, height, channels,
            devConvKernel, kernelSize, padSize
    );

    // Define global and local work sizes
    size_t globalWorkSize[3] = {
            static_cast<size_t>(width),
            static_cast<size_t>(height),
            static_cast<size_t>(channels)
    };

    // Execute the OpenCL kernel
    CLKernelEnqueue(
            queue, kernel,
            3, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(height, width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            data_size
    );

    // Free OpenCL resources

    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDst);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Output Image", result);
    cv::waitKey(0);
}

int main() {
//    KHM::sayHello();

    cl_device_id device = UserSelectDevice();

    auto max_work_group_size = CLGetInfoMaxWorkGroupSize(device);

    cl_context context =
            CLCreateContext(device);

    // demo
    conv_demo(context, device);

    clReleaseContext(context);

    return 0;
}
