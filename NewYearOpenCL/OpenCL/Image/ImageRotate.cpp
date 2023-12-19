// Image Rotate
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageRotate.h"

#include "../Kernel/KernelImageRotate.h"

OpenCLProgram CLCreateProgram_Image_Rotate(cl_context context, cl_device_id device) {
    return {
        context,
        device,
        "rotateImage",
        cl_kernel_rotate
    };
}

void KernelSetArg_Image_Rotate(
    cl_kernel kernel,
    cl_mem devSrc,
    cl_mem devDst,
    int input_width,
    int input_height,
    int channels,
    float angle
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &angle);
}
