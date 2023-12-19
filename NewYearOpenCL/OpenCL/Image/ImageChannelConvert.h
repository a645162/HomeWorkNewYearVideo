// Image Channel Convert
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_CHANNEL_CONVERT_H
#define NEW_YEAR_OPENCL_IMAGE_CHANNEL_CONVERT_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Channel(cl_context context, cl_device_id device);

void KernelSetArg_Image_Channel(
    cl_kernel kernel,
    cl_mem device_image_input,
    cl_mem device_image_output,
    int image_width, int image_height,
    int src_channels, int dst_channels
);

#endif //NEW_YEAR_OPENCL_IMAGE_CHANNEL_CONVERT_H
