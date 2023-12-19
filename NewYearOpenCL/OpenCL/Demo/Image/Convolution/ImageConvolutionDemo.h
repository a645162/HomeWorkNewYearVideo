// Demo:Image Convolution
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_IMAGE_CONVOLUTION_DEMO_H
#define NEW_YEAR_OPENCL_IMAGE_CONVOLUTION_DEMO_H

#include "../../../../OpenCV/Include/OpenCVInclude.h"

#include "../../../Include/OpenCLInclude.h"
#include "../../../Include/OpenCLError.h"
#include "../../../Include/OpenCLWorkFlow.h"
#include "../../../Include/RAII/OpenCLProgram.h"

void conv_demo(cl_context context, cl_device_id device);

#endif //NEW_YEAR_OPENCL_IMAGE_CONVOLUTION_DEMO_H
