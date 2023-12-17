// Image Rotate
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_ROTATE_H
#define NEW_YEAR_OPENCL_IMAGE_ROTATE_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLFlow.h"
#include "../Include/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Rotate(cl_context context, cl_device_id device);

void KernelSetArg_Image_Rotate(
        cl_kernel kernel,
        cl_mem devSrc,
        cl_mem devDst,
        int input_width,
        int input_height,
        int channels,
        float angle
);

#endif //NEW_YEAR_OPENCL_IMAGE_ROTATE_H
