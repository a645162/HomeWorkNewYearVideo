// Demo:Draw a Rectangle on Image
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "DrawRectDemo.h"

#include "../../../Image/Draw/DrawRect.h"

void draw_rect_demo(cl_context context, cl_device_id device) {

    cv::Mat image1 = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
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

    OpenCLProgram program_draw_rect = CLCreateProgram_Draw_Rect(context, device);

    cl_mem device_image1 = OpenCLMalloc(
            context,
            width * height * channels * sizeof(uchar),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            image1.data
    );

    cl_kernel kernel = program_draw_rect.CreateKernel();

    float frequency = 0.02;

    KernelSetArg_Draw_Rect(
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

