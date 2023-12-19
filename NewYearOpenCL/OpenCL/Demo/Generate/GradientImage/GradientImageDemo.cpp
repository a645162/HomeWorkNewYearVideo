// Demo:Gradient Color Image Generate
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "GradientImageDemo.h"

#include "../../../Image/Generate/GenerateGradientImage.h"

void gradient_image_demo(cl_context context, cl_device_id device) {
    // Read input image
    const int image_width = 1600, image_height = 1200;
    const int channels = 3;
    const int color_count = 512;

    int center_x = 200;
    int center_y = 100;

    int startR = 0, startG = 0, startB = 100;
    int endR = 0, endG = 200, endB = 200;

    auto maxR = std::max(
        {
            sqrtf(
                powf(static_cast<float>(center_x - 0), 2)
                + powf(static_cast<float>(center_y - 0), 2)
            ),
            sqrtf(
                powf(static_cast<float>(center_x - static_cast<int>(image_width)), 2)
                + powf(static_cast<float>(center_y - 0), 2)
            ),
            sqrtf(
                powf(static_cast<float>(center_x - 0), 2)
                + powf(static_cast<float>(center_y - static_cast<int>(image_height)), 2)
            ),
            sqrtf(
                powf(static_cast<float>(center_x - static_cast<int>(image_width)), 2)
                + powf(static_cast<float>(center_y - static_cast<int>(image_height)), 2)
            )
        }
    );

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_gradient_color = CLCreateProgram_Generate_GradientColor(context, device);
    OpenCLProgram program_gradient_iamge = CLCreateProgram_Generate_GradientImage(context, device);


    cl_mem device_color = OpenCLMalloc(
        context,
        color_count * channels * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );

    cl_mem device_image = OpenCLMalloc(
        context,
        image_width * image_height * channels * sizeof(uchar),
        CL_MEM_WRITE_ONLY,
        nullptr
    );

    //    cl_kernel kernel = CLCreateKernelImageResize(program);
    cl_kernel kernel_color = program_gradient_color.CreateKernel();
    cl_kernel kernel_image = program_gradient_iamge.CreateKernel();

    KernelSetArg_Generate_GradientColor(
        kernel_color,
        device_color,
        color_count,
        startR, startG, startB,
        endR, endG, endB,
        channels, 255
    );

    KernelSetArg_Generate_GradientImage(
        kernel_image,
        device_image,
        device_color,
        color_count,
        image_width, image_height,
        center_x, center_y, maxR,
        channels, 255
    );


    size_t color_work_size[1] = {
        static_cast<size_t>(color_count)
    };

    CLKernelEnqueue(
        queue, kernel_color,
        1, color_work_size
    );

    clFinish(queue);

    size_t globalWorkSize[2] = {
        static_cast<size_t>(image_width),
        static_cast<size_t>(image_height)
    };

    CLKernelEnqueue(
        queue, kernel_image,
        2, globalWorkSize
    );

    clFinish(queue);

    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(image_height, image_width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
        queue,
        result.data,
        device_image,
        image_width * image_height * channels * sizeof(uchar)
    );

    // Free OpenCL resources

    clReleaseMemObject(device_image);
    clReleaseMemObject(device_color);
    clReleaseKernel(kernel_image);
    clReleaseKernel(kernel_color);

    //    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Gradient Color Image", result);
    cv::waitKey(0);
}
