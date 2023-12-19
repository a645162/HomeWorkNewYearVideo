// Generate Gradient Image
// Created by Haomin Kong on 23-12-15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_GENERATE_GRADIENT_IMAGE_H
#define NEW_YEAR_OPENCL_GENERATE_GRADIENT_IMAGE_H

#include "../../Include/OpenCLInclude.h"
#include "../../Include/OpenCLError.h"
#include "../../Include/OpenCLWorkFlow.h"
#include "../../Include/RAII/OpenCLProgram.h"

#include "../../../OpenCV/Include/OpenCVInclude.h"

OpenCLProgram CLCreateProgram_Generate_GradientColor(cl_context context, cl_device_id device);

OpenCLProgram CLCreateProgram_Generate_GradientImage(cl_context context, cl_device_id device);

void KernelSetArg_Generate_GradientColor(
    cl_kernel kernel,
    cl_mem device_gradient_color,
    int color_count,
    uchar start_r, uchar start_g, uchar start_b,
    uchar end_r, uchar end_g, uchar end_b,
    uchar channels, uchar alpha
);

void KernelSetArg_Generate_GradientImage(
    cl_kernel kernel,
    cl_mem device_gradient_image,
    cl_mem device_gradient_color,
    int color_count,
    int image_width, int image_height,
    int center_x, int center_y, float max_r,
    uchar channels, uchar alpha
);

#endif //NEW_YEAR_OPENCL_GENERATE_GRADIENT_IMAGE_H
