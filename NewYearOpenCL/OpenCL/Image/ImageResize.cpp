//
// Created by konghaomin on 23-12-12.
//

#include "ImageResize.h"

#include <iostream>

#include "../../Author/Author.h"

#include "../Devices/OpenCLDevices.h"

#include "../Kernel/KernelImageResize.h"

#include <opencv2/opencv.hpp>

//cl_program CLCreateProgramImageResize(cl_context context, cl_device_id device) {
//    return CLCreateProgram(context, device, cl_kernel_resize_image);
//}
//
//cl_kernel CLCreateKernelImageResize(cl_program program) {
//    cl_int err;
//    auto kernel = clCreateKernel(program, "resizeImage", &err);
//    CHECK_CL_ERROR(err, "Failed to create kernel");
//    return kernel;
//}

OpenCLProgram CLCreateProgramImageResize(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "resizeImage",
            cl_kernel_resize_image
    };
}

void KernelSetArgImageResize(
        cl_kernel kernel,
        cl_mem devSrc,
        cl_mem devDst,
        int srcWidth,
        int srcHeight,
        int dstWidth,
        int dstHeight,
        int channels
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &srcWidth);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &srcHeight);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &dstWidth);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &dstHeight);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
}

unsigned int calculateNewHeightByNewWidth(
        unsigned int width,
        unsigned int height,
        unsigned int newWidth
) {
    return static_cast<int>(
            roundf(
                    static_cast<float>(height) * static_cast<float>(newWidth)
                    /
                    static_cast<float>(width)
            )
    );
}

unsigned int calculateNewWidthByNewHeight(
        unsigned int width,
        unsigned int height,
        unsigned int newHeight
) {
    return static_cast<int>(
            roundf(
                    static_cast<float>(width) * static_cast<float>(newHeight)
                    /
                    static_cast<float>(height)
            )
    );
}

void resize_demo(cl_context context, cl_device_id device) {

    // Read input image
    cv::Mat image3 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

    int srcWidth = image3.cols;
    int srcHeight = image3.rows;
    int channels = image3.channels();

    // Define the desired output size
    auto dstWidth = 400;
//    int dstHeight = 400;
    auto dstHeight =
            calculateNewHeightByNewWidth(srcWidth, srcHeight, dstWidth);

    cl_command_queue queue = CLCreateCommandQueue(context, device);


//    cl_program program = CLCreateProgramImageResize(context, device);

    OpenCLProgram resize_program = CLCreateProgramImageResize(context, device);

    // Create OpenCL buffers for input and output data

    cl_mem devSrc = OpenCLMalloc(
            context,
            srcWidth * srcHeight * channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image3.data
    );

    cl_mem devDst = OpenCLMalloc(
            context,
            dstWidth * dstHeight * channels * sizeof(uchar),
            CL_MEM_WRITE_ONLY,
            nullptr
    );

//    cl_kernel kernel = CLCreateKernelImageResize(program);
    cl_kernel kernel = resize_program.CreateKernel();

    KernelSetArgImageResize(
            kernel,
            devSrc, devDst,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            channels
    );

    // Define global and local work sizes
    size_t globalWorkSize[2] = {static_cast<size_t>(dstWidth), static_cast<size_t>(dstHeight)};

    // Execute the OpenCL kernel
    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(dstHeight, dstWidth, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            dstWidth * dstHeight * channels * sizeof(uchar)
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
