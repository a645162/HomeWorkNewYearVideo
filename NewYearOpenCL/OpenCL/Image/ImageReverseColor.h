// Image Reverse Color OpenCL
// Created by Haomin Kong on 23-12-25.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef OPENCL_IMAGE_REVERSE_COLOR_H
#define OPENCL_IMAGE_REVERSE_COLOR_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Reverse_Color(cl_context context, cl_device_id device);

void KernelSetArg_Image_Reverse_Color(
    cl_kernel kernel,
    cl_mem devSrc, cl_mem devDst,
    int input_width, int input_height, int channels
);

#endif //OPENCL_IMAGE_REVERSE_COLOR_H
