// Demo:Image Crop
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageCropDemo.h"

#include "../../../Image/ImageCrop.h"

void crop_demo(cl_context context, cl_device_id device) {

    // Read input image
    cv::Mat image3 = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    cv::resize(image3, image3, cv::Size(image3.cols / 4, image3.rows / 4));

    int input_width = image3.cols;
    int input_height = image3.rows;
    int channels = image3.channels();

    const int x1 = 300, x2 = 700, y1 = 100, y2 = 600;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_crop = CLCreateProgram_Image_Crop(context, device);

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

    KernelSetArg_Image_Crop(
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
