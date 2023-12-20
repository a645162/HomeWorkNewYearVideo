// Image Binaryzation
// Created by Haomin Kong on 23-12-20.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageBinaryzation.h"

#include "../Kernel/KernelImageBinaryzation.h"

OpenCLProgram CLCreateProgram_Image_Binaryzation(cl_context context, cl_device_id device) {
    return {
        context,
        device,
        "ImageBinaryzation",
        cl_kernel_image_binary
    };
}

void KernelSetArg_Image_Binaryzation(
    cl_kernel kernel,
    cl_mem device_image_input,
    cl_mem device_image_output,
    int image_width, int image_height, int image_channels,
    uchar threshold, bool reverse_color
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_input);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_output);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &threshold);

    const int reverse_color_int = reverse_color ? 1 : 0;
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &reverse_color_int);
}
