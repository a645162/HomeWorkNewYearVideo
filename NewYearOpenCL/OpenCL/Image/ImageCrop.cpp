//
// Created by 孔昊旻 on 2023/12/15.
//

#include "ImageCrop.h"

#include "../Kernel/KernelImageCrop.h"


#include <iostream>

#include "../../Author/Author.h"

#include "../Devices/OpenCLDevices.h"

#include <opencv2/opencv.hpp>

OpenCLProgram CLCreateProgramImageCrop(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "cropImage",
            cl_kernel_crop
    };
}

void KernelSetArgImageCrop(
        cl_kernel kernel,
        cl_mem devSrc,
        cl_mem devDst,
        int input_width,
        int input_height,
        int output_width,
        int output_height,
        int x1, int y1,
        int x2, int y2,
        int channels
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &output_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &output_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x2);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y2);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
}

void crop_demo(cl_context context, cl_device_id device) {

    // Read input image
    cv::Mat image3 = cv::imread("../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    cv::resize(image3, image3, cv::Size(image3.cols / 4, image3.rows / 4));

    int input_width = image3.cols;
    int input_height = image3.rows;
    int channels = image3.channels();

    const int x1 = 300, x2 = 700, y1 = 100, y2 = 600;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_crop = CLCreateProgramImageCrop(context, device);

    const int x_1 = std::min(x1, x2);
    const int y_1 = std::min(y1, y2);
    const int x_2 = std::max(x1, x2);
    const int y_2 = std::max(y1, y2);

    const int output_width = x_2 - x_1;
    const int output_height = y_2 - y_1;

    cl_mem devSrc = OpenCLMalloc(
            context,
            input_width * input_height * channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image3.data
    );

    cl_mem devDst = OpenCLMalloc(
            context,
            output_width * output_height * channels * sizeof(uchar),
            CL_MEM_WRITE_ONLY,
            nullptr
    );

//    cl_kernel kernel = CLCreateKernelImageResize(program);
    cl_kernel kernel = program_crop.CreateKernel();

    KernelSetArgImageCrop(
            kernel,
            devSrc, devDst,
            input_width, input_height,
            output_width, output_height,
            x1, y1, x2, y2,
            channels
    );

    size_t globalWorkSize[2] = {
            static_cast<size_t>(output_width),
            static_cast<size_t>(output_height)
    };

    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(output_height, output_width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            output_width * output_height * channels * sizeof(uchar)
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
