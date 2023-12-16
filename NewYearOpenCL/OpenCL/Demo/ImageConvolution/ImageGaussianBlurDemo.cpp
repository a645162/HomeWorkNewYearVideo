//
// Created by konghaomin on 23-12-16.
//

#include "ImageGaussianBlurDemo.h"

#include "../../Image/ImageConvolution.h"
#include "../../Image/GaussianKernel.h"

void blur_conv_demo(cl_context context, cl_device_id device) {

    cv::Mat image_ori = cv::imread("../../../Resources/Image/input.png", cv::IMREAD_UNCHANGED);
//    cv::Mat image3 = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);
    cv::resize(image_ori, image_ori, cv::Size(image_ori.cols / 4, image_ori.rows / 4));

    // remove alpha test
    cv::cvtColor(image_ori, image_ori, cv::COLOR_BGRA2BGR);

    // Convert to gray test
//    cv::cvtColor(image_ori, image_ori, cv::COLOR_BGR2GRAY);

    auto width = image_ori.cols;
    auto height = image_ori.rows;
    auto channels = image_ori.channels();

    std::cout << "Image size: " << width << "x" << height << "x" << channels << std::endl;

    cl_command_queue queue = CLCreateCommandQueue(context, device);

//    cl_program program = CLCreateProgramImageResize(context, device);

    OpenCLProgram program_conv = CLCreateProgram_Image_Conv(context, device);

    // Create OpenCL buffers for input and output data

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

    auto gaussian_params = calcGaussianKernelParameters(90.0f);
    auto kernel_gaussian = createGaussianKernel(gaussian_params);

    const int kernelSize = gaussian_params.size;
    const int padSize = kernelSize / 2;

    cl_mem devConvKernel1 = OpenCLMalloc(
            context,
            kernelSize * kernelSize * sizeof(float),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (void *) (kernel_gaussian)
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


    // Copy the result from OpenCL device memory back to Mat
    cv::Mat result(height, width, CV_8UC(channels));

    OpenCLMemcpyFromDevice(
            queue,
            result.data,
            devDst,
            img_data_size
    );

    // Free OpenCL resources

    clReleaseMemObject(devSrc);
    clReleaseMemObject(devDst);
    clReleaseKernel(kernel);

//    clReleaseProgram(resize_program);

    clReleaseCommandQueue(queue);

    cv::imshow("Input Image", image_ori);
    cv::imshow("Output Image", result);
    cv::waitKey(0);
}
