// Image Crop
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_CROP_H
#define NEW_YEAR_OPENCL_IMAGE_CROP_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Crop(cl_context context, cl_device_id device);

void KernelSetArg_Image_Crop(
    cl_kernel kernel,
    cl_mem devSrc,
    cl_mem devDst,
    int input_width, int input_height,
    int output_width, int output_height,
    int x1, int y1,
    int x2, int y2,
    int channels
);

#endif //NEW_YEAR_OPENCL_IMAGE_CROP_H
