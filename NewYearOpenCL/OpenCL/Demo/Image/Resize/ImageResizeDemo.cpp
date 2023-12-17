// Demo:Image Resize
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageResizeDemo.h"

#include "../../../Image/ImageResize.h"

void resize_demo(cl_context context, cl_device_id device) {

    // Read input image
    cv::Mat image3 = cv::imread("../../../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

    int srcWidth = image3.cols;
    int srcHeight = image3.rows;
    int channels = image3.channels();

    // Define the desired output size
    auto dstWidth = 400;
//    dstWidth = 4000;
//    int dstHeight = 400;
    auto dstHeight =
            calculateNewHeightByNewWidth(srcWidth, srcHeight, dstWidth);

    cl_command_queue queue = CLCreateCommandQueue(context, device);


//    cl_program program = CLCreateProgramImageResize(context, device);

    OpenCLProgram resize_program = CLCreateProgram_Image_Resize(context, device);

    // Create OpenCL buffers for input and output data

    cl_mem devSrc = OpenCLMalloc(
            context,
            srcWidth * srcHeight * channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image3.data
    );

    cl_mem devDst = OpenCLMalloc(
            context,
            dstWidth * dstHeight * channels * sizeof(uchar),
            CL_MEM_WRITE_ONLY,
            nullptr
    );

//    cl_kernel kernel = CLCreateKernelImageResize(program);
    cl_kernel kernel = resize_program.CreateKernel();

    KernelSetArg_Image_Resize(
            kernel,
            devSrc, devDst,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            channels
    );

    // Define global and local work sizes
    size_t globalWorkSize[2] = {static_cast<size_t>(dstWidth), static_cast<size_t>(dstHeight)};

    // Execute the OpenCL kernel
    CLKernelEnqueue(
            queue, kernel,
            2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(dstHeight, dstWidth, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            dstWidth * dstHeight * channels * sizeof(uchar)
    );

    // Free OpenCL resources

    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDst);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Resized Image", result);
    cv::waitKey(0);
}
