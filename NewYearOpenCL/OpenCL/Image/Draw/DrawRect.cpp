//
// Created by konghaomin on 23-12-16.
//

#include "DrawRect.h"

#include "../../Kernel/KernelDrawRect.h"

#include <iostream>

#include <opencv2/opencv.hpp>

OpenCLProgram CLCreateProgramDrawRect(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "drawRectangle",
            cl_kernel_draw_rect
    };
}

void KernelSetArgDrawRect(
        cl_kernel kernel,
        cl_mem device_image,
        int width, int height,
        int x1, int y1,
        int x2, int y2,
        int thickness,
        uchar board_color_r, uchar board_color_g, uchar board_color_b,
        uchar fill_color_r, uchar fill_color_g, uchar fill_color_b,
        int channels, int fill,
        int sine_waves_board, float frequency
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y1);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x2);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y2);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &thickness);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &board_color_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &board_color_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &board_color_b);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &fill_color_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &fill_color_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &fill_color_b);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &fill);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &sine_waves_board);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &frequency);
}

void draw_rect_demo(cl_context context, cl_device_id device) {

    cv::Mat image1 = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
//    cv::Mat image2 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

//    cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
//    cv::cvtColor(image1, image1, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(image1, image1, cv::COLOR_BGRA2BGR);

    cv::resize(image1, image1, cv::Size(image1.cols / 4, image1.rows / 4));
    int width = image1.cols;
    int height = image1.rows;
    int channels = image1.channels();
    std::cout << width << "x" << height << "x" << channels << std::endl;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_draw_rect = CLCreateProgramDrawRect(context, device);

    cl_mem device_image1 = OpenCLMalloc(
            context,
            width * height * channels * sizeof(uchar),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            image1.data
    );

    cl_kernel kernel = program_draw_rect.CreateKernel();

    float frequency = 0.02;

    KernelSetArgDrawRect(
            kernel,
            device_image1,
            width, height,
            100, 100,
            700, 500,
            10,
            255, 255, 255,
            0, 0, 255,
            channels,
            true,
            true,
            frequency
    );

    size_t globalWorkSize[2] = {
            static_cast<size_t>(width),
            static_cast<size_t>(height)
    };

    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(height, width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_image1,
            width * height * channels * sizeof(uchar)
    );

    // Free OpenCL resources

    clReleaseMemObject(device_image1);
    clReleaseKernel(kernel);

    clReleaseCommandQueue(queue);

    std::cout << "Output:" << std::endl;
    std::cout << result.cols << "x" << result.rows << "x" << result.channels() << std::endl;
    cv::imshow("Output Image", result);
    cv::waitKey(0);
}

