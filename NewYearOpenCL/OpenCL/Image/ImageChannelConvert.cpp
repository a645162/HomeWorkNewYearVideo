// Image Channel Convert
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageChannelConvert.h"

#include "../Kernel/KernelImageChannel.h"

#include <iostream>


#include "../Devices/OpenCLDevices.h"

#include <opencv2/opencv.hpp>

OpenCLProgram CLCreateProgramImageChannel(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "ImageChannelConvert",
            cl_kernel_channel
    };
}

void KernelSetArgImageChannel(
        cl_kernel kernel,
        cl_mem device_image_input,
        cl_mem device_image_output,
        int image_width, int image_height,
        int src_channels, int dst_channels
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_input);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_output);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &src_channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &dst_channels);
}

void convert_channel_demo(cl_context context, cl_device_id device) {

    cv::Mat image1 = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
//    cv::Mat image2 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

//    cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
//    cv::cvtColor(image1, image1, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(image1, image1, cv::COLOR_BGRA2BGR);

    cv::resize(image1, image1, cv::Size(image1.cols / 4, image1.rows / 4));
    int image1_width = image1.cols;
    int image1_height = image1.rows;
    int image1_channels = image1.channels();
    std::cout << image1_width << "x" << image1_height << "x" << image1_channels << std::endl;

    const unsigned int output_channel = 4;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_channel = CLCreateProgramImageChannel(context, device);

    cl_mem device_image1 = OpenCLMalloc(
            context,
            image1_width * image1_height * image1_channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image1.data
    );

    cl_mem device_output = OpenCLMalloc(
            context,
            image1_width * image1_height * output_channel * sizeof(uchar),
            CL_MEM_WRITE_ONLY,
            nullptr
    );

    cl_kernel kernel = program_channel.CreateKernel();

    KernelSetArgImageChannel(
            kernel,
            device_image1, device_output,
            image1_width, image1_height,
            image1_channels,
            output_channel
    );

    size_t globalWorkSize[2] = {
            static_cast<size_t>(image1_width),
            static_cast<size_t>(image1_height)
    };

    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(image1_height, image1_width, CV_8UC(output_channel));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_output,
            image1_width * image1_height * output_channel * sizeof(uchar)
    );

    // Free OpenCL resources

    clReleaseMemObject(device_image1);
    clReleaseMemObject(device_output);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    std::cout << "Output:" << std::endl;
    std::cout << result.cols << "x" << result.rows << "x" << result.channels() << std::endl;
    cv::imshow("Output Image", result);
    cv::waitKey(0);
}

