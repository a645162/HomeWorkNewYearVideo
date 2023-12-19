// Image Convolution on GPU
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_CONVOLUTION_H
#define NEW_YEAR_OPENCL_IMAGE_CONVOLUTION_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Conv(cl_context context, cl_device_id device);

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
);

#endif //NEW_YEAR_OPENCL_IMAGE_CONVOLUTION_H
