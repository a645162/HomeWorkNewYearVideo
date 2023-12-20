// Demo:Image Convolution
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageConvolutionDemo.h"

#include "../../../Image/ImageConvolution.h"
#include "../../../Image/ImageBinaryzation.h"

void conv_demo(cl_context context, cl_device_id device) {
    cv::Mat image_ori = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
    //    cv::Mat image3 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);
    cv::resize(image_ori, image_ori, cv::Size(image_ori.cols / 4, image_ori.rows / 4));

    // remove alpha
    //    cv::cvtColor(image_ori, image_ori, cv::COLOR_BGRA2BGR);

    // Convert to gray
    //    cv::cvtColor(image_ori, image_ori, cv::COLOR_BGR2GRAY);

    cv::cvtColor(image_ori, image_ori, cv::COLOR_BGRA2BGR);

    auto width = image_ori.cols;
    auto height = image_ori.rows;
    auto channels = image_ori.channels();

    std::cout << "Image size: " << width << "x" << height << "x" << channels << std::endl;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_conv = CLCreateProgram_Image_Conv(context, device);

    // Calculate Image Data Size
    const auto img_data_size = width * height * channels * sizeof(uchar);

    cl_mem devSrc = OpenCLMalloc(
        context,
        img_data_size,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        image_ori.data
    );

    cl_mem devDst = OpenCLMalloc(
        context,
        img_data_size,
        CL_MEM_READ_WRITE,
        nullptr
    );

    cl_mem devDst1 = OpenCLMalloc(
        context,
        img_data_size,
        CL_MEM_WRITE_ONLY,
        nullptr
    );

    const float kernel_laplacian[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};

    const float kernel_ori[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};

    const float kernel_x[] = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };
    const float kernel_y[] = {
        1, 1, 1,
        0, 0, 0,
        -1, -1, -1
    };
    const int kernelSize = 3;
    const int padSize = kernelSize / 2;

    cl_mem devConvKernel1 = OpenCLMalloc(
        context,
        kernelSize * kernelSize * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (void *) (kernel_x)
    );

    cl_mem devConvKernel2 = OpenCLMalloc(
        context,
        kernelSize * kernelSize * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (void *) (kernel_y)
    );

    cl_kernel kernel = program_conv.CreateKernel();

    KernelSetArg_Image_Conv(
        kernel,
        devSrc, devDst,
        height, width, channels,
        devConvKernel1, kernelSize, padSize
    );

    size_t globalWorkSize[3] = {
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        static_cast<size_t>(channels)
    };

    CLKernelEnqueue(
        queue, kernel,
        3, globalWorkSize
    );

    clFinish(queue);

    cl_kernel kernel1 = program_conv.CreateKernel();

    KernelSetArg_Image_Conv(
        kernel1,
        devDst, devDst1,
        height, width, channels,
        devConvKernel2, kernelSize, padSize
    );

    CLKernelEnqueue(
        queue, kernel1,
        3, globalWorkSize
    );

    clFinish(queue);

    auto program_binaryzation = CLCreateProgram_Image_Binaryzation(context, device);

    const auto mem_binaryzation = OpenCLMem(
        context,
        width, height, channels
    );

    const auto kernel_binaryzation = program_binaryzation.CreateKernelRAII();
    KernelSetArg_Image_Binaryzation(
        kernel_binaryzation,
        devDst1, mem_binaryzation,
        width, height, channels,
        100, false
    );
    size_t globalWorkSize_2[2] = {
        static_cast<size_t>(width),
        static_cast<size_t>(height)
    };
    kernel_binaryzation.Execute(queue, 2, globalWorkSize_2);


    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(height, width, CV_8UC(channels));
    cv::Mat result_bin(height, width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
        queue,
        result.data,
        devDst1,
        img_data_size
    );

    mem_binaryzation.CopyToHost(queue, result_bin.data);

    // Free OpenCL resources

    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDst);
    clReleaseKernel(kernel);

    //    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Input Image", image_ori);
    cv::imshow("Output Image", result);
    cv::imshow("Output Bin Image", result_bin);
    cv::waitKey(0);
}
