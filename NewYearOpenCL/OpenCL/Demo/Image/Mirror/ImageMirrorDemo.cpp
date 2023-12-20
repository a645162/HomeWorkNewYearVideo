// Demo:Image Mirror
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageMirrorDemo.h"

#include "../../../Include/RAII/OpenCLProgram.h"
#include "../../../Include/RAII/OpenCLQueue.h"
#include "../../../Include/RAII/OpenCLKernel.h"
#include "../../../Include/RAII/OpenCLMem.h"

#include "../../../../Utils/Calc.h"

#include "../../../Image/ImageMirror.h"

void mirror_demo(cl_context context, cl_device_id device) {
    cv::Mat image1 = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);

    //    cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
    //    cv::cvtColor(image1, image1, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(image1, image1, cv::COLOR_BGRA2BGR);

    cv::resize(image1, image1, cv::Size(image1.cols / 4, image1.rows / 4));
    const int image1_width = image1.cols;
    const int image1_height = image1.rows;
    const int image1_channels = image1.channels();
    std::cout << image1_width << "x" << image1_height << "x" << image1_channels << std::endl;

    const auto queue = OpenCLQueue(context, device);

    OpenCLProgram program_mirror = CLCreateProgram_Image_Mirror(context, device);

    const auto device_image1 = OpenCLMem(
        context,
        calcImageSize(image1_width, image1_height, image1_channels),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        image1.data
    );

    const auto device_output = OpenCLMem(
        context,
        image1_width, image1_height, image1_channels
    );

    const auto kernel = program_mirror.CreateKernelRAII();

    KernelSetArg_Image_Mirror(
        kernel,
        device_image1, device_output,
        image1_width, image1_height,
        image1_channels
    );

    KernelSetArg_Image_Mirror(
        kernel,
        device_image1, device_output,
        image1_width, image1_height,
        image1_channels,
        0
    );

    size_t globalWorkSize[2] = {
        static_cast<size_t>(image1_width),
        static_cast<size_t>(image1_height)
    };

    CLKernelEnqueue(
        queue, kernel,
        2, globalWorkSize
    );

    queue.WaitFinish();

    const cv::Mat result(image1_height, image1_width, CV_8UC(image1_channels));

    device_output.CopyToHost(queue, result.data);

    std::cout << "Output:" << std::endl;
    std::cout << result.cols << "x" << result.rows << "x" << result.channels() << std::endl;

    cv::imshow("Output Image", result);
    cv::waitKey(0);
}
