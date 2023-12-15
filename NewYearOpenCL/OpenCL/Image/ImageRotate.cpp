//
// Created by 孔昊旻 on 2023/12/15.
//

#include "ImageRotate.h"

#include "../Kernel/KernelImageRotate.h"

#include <iostream>

#include "../../Author/Author.h"

#include "../Devices/OpenCLDevices.h"

#include <opencv2/opencv.hpp>

OpenCLProgram CLCreateProgramImageRotate(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "rotateImage",
            cl_kernel_rotate
    };
}

void KernelSetArgImageRotate(
        cl_kernel kernel,
        cl_mem devSrc,
        cl_mem devDst,
        int input_width,
        int input_height,
        int channels,
        float angle
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &angle);
}

void rotate_demo(cl_context context, cl_device_id device) {

    cv::Mat image3 = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    cv::resize(image3, image3, cv::Size(image3.cols / 4, image3.rows / 4));

    int input_width = image3.cols;
    int input_height = image3.rows;
    int channels = image3.channels();

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_rotate = CLCreateProgramImageRotate(context, device);

    cl_mem devSrc = OpenCLMalloc(
            context,
            input_width * input_height * channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image3.data
    );

    cl_mem devDst = OpenCLMalloc(
            context,
            input_width * input_height * channels * sizeof(uchar),
            CL_MEM_WRITE_ONLY,
            nullptr
    );

    cl_kernel kernel = program_rotate.CreateKernel();

    KernelSetArgImageRotate(
            kernel,
            devSrc, devDst,
            input_width, input_height,
            channels,
            -45.0f
    );

    size_t globalWorkSize[2] = {
            static_cast<size_t>(input_width),
            static_cast<size_t>(input_height)
    };

    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(input_height, input_width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            input_width * input_height * channels * sizeof(uchar)
    );

    // Free OpenCL resources

    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDst);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Croped Image", result);
    cv::waitKey(0);
}
