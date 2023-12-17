// Demo:Image Rotate
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageRotateDemo.h"

#include "../../../Image/ImageRotate.h"

void rotate_demo(cl_context context, cl_device_id device) {

    cv::Mat image3 = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    cv::resize(image3, image3, cv::Size(image3.cols / 4, image3.rows / 4));

    int input_width = image3.cols;
    int input_height = image3.rows;
    int channels = image3.channels();

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_rotate = CLCreateProgram_Image_Rotate(context, device);

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

    KernelSetArg_Image_Rotate(
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