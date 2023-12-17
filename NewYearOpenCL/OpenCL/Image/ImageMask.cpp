// Image Mask
// Created by Haomin Kong on 23-12-12.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageMask.h"

#include "../../Config/Path.h"
#include "../../Config/DebugVar.h"

#include "../Kernel/KernelMaskImage.h"

#include <iostream>

#include <opencv2/opencv.hpp>

#include "../Devices/OpenCLDevices.h"


[[maybe_unused]] typedef struct {
    uchar x;
    uchar y;
    uchar z;
} uchar3;

void processImageOpenCL(
        cl_device_id device,
        cl_context context,
        unsigned char *h_input, unsigned char *h_output,
        int width, int height, int channels,
        int centerX, int centerY,
        float radius
) {
    cl_int err;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_mask = OpenCLProgram(
            context, device,
            "MaskImageCircle",
            cl_kernel_mask_image_circle
    );

    cl_kernel kernel = program_mask.CreateKernel();

    cl_mem d_input = clCreateBuffer(
            context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            width * height * channels * sizeof(unsigned char), h_input, &err
    );
    CHECK_CL_ERROR(err, "clCreateBuffer");

    cl_mem d_output = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, width * height * channels * sizeof(unsigned char),
            nullptr, &err
    );
    CHECK_CL_ERROR(err, "clCreateBuffer");

//    int centerX = 2 * width / 3, centerY = height / 2;

    const auto light_source_x = (width / 2);
    const auto light_source_y = -100;

    const auto k_center =
            (static_cast<float>(centerY - light_source_y)) / (static_cast<float>(centerX - light_source_x));
    const auto angle_center = atanf(k_center);
    const auto distance_center = sqrtf(
            powf(static_cast<float>(light_source_x - centerX), 2) +
            powf(static_cast<float>(light_source_y - centerY), 2)
    );
    const auto max_distance = sqrtf(
            powf(distance_center, 2) - powf(radius, 2)
    );

    const auto angle_between_center = asinf(radius / distance_center);
    const auto angle_1 = angle_center - angle_between_center;
    const auto angle_2 = angle_center + angle_between_center;

    const auto m_1 = 1.0f / tanf(angle_1);
    const auto m_2 = 1.0f / tanf(angle_2);

    // 设置内核参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    CHECK_CL_ERROR(err, "clSetKernelArg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    CHECK_CL_ERROR(err, "clSetKernelArg");

    err = clSetKernelArg(kernel, 2, sizeof(int), &width);
    err = clSetKernelArg(kernel, 3, sizeof(int), &height);
    err = clSetKernelArg(kernel, 4, sizeof(int), &channels);

    err = clSetKernelArg(kernel, 5, sizeof(int), &centerX);
    err = clSetKernelArg(kernel, 6, sizeof(int), &centerY);
    err = clSetKernelArg(kernel, 7, sizeof(float), &radius);

    int clean_up_alpha = 1;
    int focus_lamp = 1;
    err = clSetKernelArg(kernel, 8, sizeof(int), &clean_up_alpha);
    err = clSetKernelArg(kernel, 9, sizeof(int), &focus_lamp);

    err = clSetKernelArg(kernel, 10, sizeof(int), &light_source_x);
    err = clSetKernelArg(kernel, 11, sizeof(int), &light_source_y);

    err = clSetKernelArg(kernel, 12, sizeof(float), &m_1);
    err = clSetKernelArg(kernel, 13, sizeof(float), &m_2);
    err = clSetKernelArg(kernel, 14, sizeof(float), &max_distance);

    uchar c = 0, alpha = 150;

    err = clSetKernelArg(kernel, 15, sizeof(uchar), &c);
    err = clSetKernelArg(kernel, 16, sizeof(uchar), &c);
    err = clSetKernelArg(kernel, 17, sizeof(uchar), &c);

    err = clSetKernelArg(kernel, 18, sizeof(uchar), &alpha);

    // 执行内核
    size_t global_size[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel");


    clFinish(queue);

    // 从设备端读取结果到主机端
    err = clEnqueueReadBuffer(
            queue, d_output, CL_TRUE, 0, width * height * channels * sizeof(unsigned char), h_output,
            0, nullptr, nullptr
    );
    CHECK_CL_ERROR(err, "clEnqueueReadBuffer");

    // 释放OpenCL资源
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);

    clReleaseCommandQueue(queue);

}

void mask_video_demo(cl_context context, cl_device_id device) {

    cv::Mat image = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    cv::resize(image, image, cv::Size(1080, 607));
//    cv::imshow("Input", image);
//    cv::waitKey(0);

    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        exit(1);
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    const float radius = 150.5f;  // Set your desired radius
//    int centerX = 2 * width / 3;
    int centerY = height / 2;

    uchar *h_input = image.data;

    int totalFrames = 300;

//    cv::VideoWriter writer(
//            "focus_video.avi",
//            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
//            60,
//            cv::Size(width, height)
//    );
    cv::VideoWriter writer(
            "focus_video.mp4",
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            60,
            cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Error: Couldn't create the output video file." << std::endl;
        exit(1);
    }

    // Save h_output as your result image
//    cv::Mat result(height, width, CV_8UC4);
//
//    processImageOpenCL(
//            h_input, result.data,
//            width, height, channels,
//            centerX, centerY,
//            radius
//    );

    for (int frameIdx = 0; frameIdx < totalFrames; ++frameIdx) {

        cv::Mat result(height, width, CV_8UC4);

        // 设置centerX从0到width的变化
//        int centerX = frameIdx % width;
        int centerX = static_cast<int>(
                static_cast<float>(width)
                *
                (static_cast<float>(frameIdx) / static_cast<float>(totalFrames))
        );

        processImageOpenCL(
                device,
                context,
                h_input, result.data, width, height, channels, centerX, centerY, radius
        );

        // convert to 3 channel
        cv::cvtColor(result, result, cv::COLOR_BGRA2BGR);

        writer.write(result);

        cv::imshow("Output", result);
        cv::waitKey(1);
    }


//    cv::imshow("Input", image);
//    cv::imshow("Output", result);
//    cv::waitKey(0);
}
