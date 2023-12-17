// Image Resize
// Created by Haomin Kong on 23-12-12.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_RESIZE_IMAGE_H
#define NEW_YEAR_OPENCL_RESIZE_IMAGE_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLFlow.h"
#include "../Include/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Resize(cl_context context, cl_device_id device);

void KernelSetArg_Image_Resize(
        cl_kernel kernel,
        cl_mem devSrc, cl_mem devDst,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        int channels
);

[[maybe_unused]] unsigned int calculateNewHeightByNewWidth(
        unsigned int width,
        unsigned int height,
        unsigned int newWidth
);

[[maybe_unused]] unsigned int calculateNewWidthByNewHeight(
        unsigned int width,
        unsigned int height,
        unsigned int newHeight
);

#endif //NEW_YEAR_OPENCL_RESIZE_IMAGE_H
