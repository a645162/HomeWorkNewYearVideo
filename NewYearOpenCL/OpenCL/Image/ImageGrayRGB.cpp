// Image Gray
// Created by Haomin Kong on 23-12-15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageGrayRGB.h"

#include "../Kernel/KernelImageGrayRGB.h"

OpenCLProgram CLCreateProgram_Image_Gray_RGB(cl_context context, cl_device_id device) {
    return {
        context,
        device,
        "convertToGrayRGB",
        cl_kernel_gray
    };
}

void KernelSetArg_Image_Gray_RGB(
    cl_kernel kernel,
    cl_mem device_image_input,
    cl_mem device_image_output,
    int image_width, int image_height, int channels,
    int type
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_input);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_output);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &type);
}
