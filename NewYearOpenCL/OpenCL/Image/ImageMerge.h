// Image Merge
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_MERGE_H
#define NEW_YEAR_OPENCL_IMAGE_MERGE_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Merge(cl_context context, cl_device_id device);

void KernelSetArg_Image_Merge(
    cl_kernel kernel,
    cl_mem image1,
    cl_mem image2,
    cl_mem device_output,
    int image1_width, int image1_height, int image1_channels,
    int image2_target_x, int image2_target_y,
    int image2_width, int image2_height, int image2_channels,
    int image2_alpha
);

#endif //NEW_YEAR_OPENCL_IMAGE_MERGE_H
