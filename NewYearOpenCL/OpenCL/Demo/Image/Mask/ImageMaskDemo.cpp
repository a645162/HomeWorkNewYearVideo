// Demo:Image Mask
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageMaskDemo.h"

#include "../../../Image/ImageMask.h"

void processImageMaskOpenCL(
    cl_device_id device,
    cl_context context,
    unsigned char* h_input, unsigned char* h_output,
    int width, int height, int channels,
    int centerX, int centerY,
    float radius
) {
    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_mask = CLCreateProgram_Image_Mask(context, device);

    cl_kernel kernel = program_mask.CreateKernel();

    cl_mem d_input = OpenCLMalloc(
        context,
        width * height * channels * sizeof(unsigned char),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        h_input
    );

    cl_mem d_output = OpenCLMalloc(
        context,
        width * height * channels * sizeof(unsigned char),
        CL_MEM_WRITE_ONLY,
        nullptr
    );

    //    int centerX = 2 * width / 3, centerY = height / 2;

    const auto light_source_x = (width / 2);
    const auto light_source_y = -100;

    KernelSetArg_Image_Mask_Simple(
        kernel,
        d_input, d_output,
        width, height, channels,
        centerX, centerY, radius,
        1, 1,
        light_source_x, light_source_y,
        0, 0, 0, 150
    );


    size_t global_size[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};

    CLKernelEnqueue(
        queue, kernel,
        2, global_size
    );

    clFinish(queue);

    // Copy from device to host
    OpenCLMemcpyFromDevice(
        queue,
        h_output,
        d_output,
        width * height * channels * sizeof(unsigned char)
    );

    // Release
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);

    clReleaseCommandQueue(queue);
}

void mask_video_demo(cl_context context, cl_device_id device) {
    cv::Mat image = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
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
    const float radius = 150.5f; // Set your desired radius
    //    int centerX = 2 * width / 3;
    int centerY = height / 2;

    uchar* h_input = image.data;

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

        processImageMaskOpenCL(
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
