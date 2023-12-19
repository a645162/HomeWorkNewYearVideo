// Image Gray
// Created by Haomin Kong on 23-12-15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_GRAY_RGB_H
#define NEW_YEAR_OPENCL_IMAGE_GRAY_RGB_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Gray_RGB(cl_context context, cl_device_id device);

void KernelSetArg_Image_Gray_RGB(
    cl_kernel kernel,
    cl_mem device_image_input,
    cl_mem device_image_output,
    int image_width, int image_height, int channels,
    int type = 1
);

#endif //NEW_YEAR_OPENCL_IMAGE_GRAY_RGB_H
