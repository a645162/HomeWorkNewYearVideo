// Image Merge
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageMerge.h"

#include "../Kernel/KernelImageMerge.h"

OpenCLProgram CLCreateProgram_Image_Merge(cl_context context, cl_device_id device) {
    return {
        context,
        device,
        "mergeImages",
        cl_kernel_merge
    };
}

void KernelSetArg_Image_Merge(
    cl_kernel kernel,
    cl_mem image1,
    cl_mem image2,
    cl_mem device_output,
    int image1_width, int image1_height, int image1_channels,
    int image2_target_x, int image2_target_y,
    int image2_width, int image2_height, int image2_channels,
    int image2_alpha
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &image1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &image2);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_output);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image1_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image1_height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image1_channels);

    // Target Position
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image2_target_x);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image2_target_y);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image2_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image2_height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image2_channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image2_alpha);
}
