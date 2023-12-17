// Demo:Image Merge
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageMergeDemo.h"

#include "../../../Image/ImageMerge.h"

void merge_demo(cl_context context, cl_device_id device) {

    std::cout << "Image Merge Demo" << std::endl;

    cv::Mat image1 = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("../../../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

    cv::resize(image1, image1, cv::Size(image1.cols / 4, image1.rows / 4));
    cv::resize(image2, image2, cv::Size(image2.cols / 5, image2.rows / 5));

    // Test on 3 channel
//    cv::cvtColor(image1, image1, cv::COLOR_BGRA2BGR);
//    cv::cvtColor(image2, image2, cv::COLOR_BGRA2BGR);

    int image1_width = image1.cols;
    int image1_height = image1.rows;
    int image1_channels = image1.channels();
    std::cout << image1_width << "x" << image1_height << "x" << image1_channels << std::endl;


    int image2_width = image2.cols;
    int image2_height = image2.rows;
    int image2_channels = image2.channels();
    std::cout << image2_width << "x" << image2_height << "x" << image2_channels << std::endl;

    int target_x = -100, target_y = 100;


    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_merge = CLCreateProgram_Image_Merge(context, device);

    cl_mem device_image1 = OpenCLMalloc(
            context,
            image1_width * image1_height * image1_channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image1.data
    );

    cl_mem device_image2 = OpenCLMalloc(
            context,
            image2_width * image2_height * image2_channels * sizeof(uchar),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            image2.data
    );

    cl_mem device_output = OpenCLMalloc(
            context,
            image1_width * image1_height * image1_channels * sizeof(uchar),
            CL_MEM_WRITE_ONLY,
            nullptr
    );

    cl_kernel kernel = program_merge.CreateKernel();

    KernelSetArg_Image_Merge(
            kernel,
            device_image1, device_image2, device_output,
            image1_width, image1_height, image1_channels,
            target_x, target_y,
            image2_width, image2_height, image2_channels,
            100
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
    cv::Mat result(image1_height, image1_width, CV_8UC(image1_channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_output,
            image1_width * image1_height * image1_channels * sizeof(uchar)
    );

    // Free OpenCL resources

    clReleaseMemObject(device_image1);
    clReleaseMemObject(device_image2);
    clReleaseMemObject(device_output);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Croped Image", result);
    cv::waitKey(0);
}
