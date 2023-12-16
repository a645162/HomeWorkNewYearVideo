// Image Convolution on GPU
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageConvolution.h"

#include "../Kernel/KernelImageConvolution.h"

OpenCLProgram CLCreateProgram_Image_Conv(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "convolution2Dim",
            cl_kernel_convolution
    };
}

void KernelSetArg_Image_Conv(
        cl_kernel kernel,
        cl_mem device_src,
        cl_mem device_dst,
        int height,
        int width,
        int channels,
        cl_mem conv_kernel,
        int conv_kernel_size,
        int padSize
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_src);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_dst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &conv_kernel);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &conv_kernel_size);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &padSize);
}