// Image Binaryzation
// Created by Haomin Kong on 23-12-20.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_BINARYZATION_H
#define NEW_YEAR_OPENCL_IMAGE_BINARYZATION_H

#include "../../OpenCV/Include/OpenCVInclude.h"

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"

#include "../Include/OpenCLRAII.h"

#define WORK_DIM_IMAGE_BINARYZATION 2

OpenCLProgram CLCreateProgram_Image_Binaryzation(cl_context context, cl_device_id device);

void KernelSetArg_Image_Binaryzation(
    cl_kernel kernel,
    cl_mem device_image_input,
    cl_mem device_image_output,
    int image_width, int image_height, int image_channels,
    uchar threshold, bool reverse_color = false
);

#endif //NEW_YEAR_OPENCL_IMAGE_BINARYZATION_H
