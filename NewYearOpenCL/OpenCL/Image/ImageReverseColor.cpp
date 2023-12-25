// Image Reverse Color OpenCL
// Created by Haomin Kong on 23-12-25.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageReverseColor.h"

#include "../Kernel/KernelImageReverseColor.h"

OpenCLProgram CLCreateProgram_Image_Reverse_Color(cl_context context, cl_device_id device)
{
    return {
        context,
        device,
        "ImageReverseColor",
        cl_kernel_reverse_color
    };
}

void KernelSetArg_Image_Reverse_Color(
    cl_kernel kernel,
    cl_mem devSrc, cl_mem devDst,
    int input_width, int input_height, int channels
)
{
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
}
