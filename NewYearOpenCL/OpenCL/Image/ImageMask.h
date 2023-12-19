// Image Mask
// Created by Haomin Kong on 23-12-12.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_MASK_IMAGE_H
#define NEW_YEAR_OPENCL_MASK_IMAGE_H

#include "../../OpenCV/Include/OpenCVInclude.h"

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLWorkFlow.h"
#include "../Include/RAII/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Mask(cl_context context, cl_device_id device);

void KernelSetArg_Image_Mask(
    cl_kernel kernel,
    cl_mem device_input, cl_mem device_output,
    int width, int height, int channels,
    int centerX, int centerY, float radius,
    int clean_up_alpha, int focus_lamp,
    int light_source_x, int light_source_y,
    float m_1, float m_2, float max_distance,
    uchar focus_color_b, uchar focus_color_g, uchar focus_color_r, uchar color_alpha
);

void KernelSetArg_Image_Mask_Simple(
    cl_kernel kernel,
    cl_mem device_input, cl_mem device_output,
    int width, int height, int channels,
    int centerX, int centerY, float radius,
    int clean_up_alpha, int focus_lamp,
    int light_source_x, int light_source_y,
    uchar focus_color_b, uchar focus_color_g, uchar focus_color_r, uchar color_alpha
);

#endif //NEW_YEAR_OPENCL_MASK_IMAGE_H
