// Project: New Year Card Video
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_NEW_YEAR_CARD_VIDEO_H
#define NEW_YEAR_OPENCL_NEW_YEAR_CARD_VIDEO_H

#include "../OpenCV/Include/OpenCVInclude.h"

#include "../OpenCL/Include/OpenCLInclude.h"
#include "../OpenCL/Include/OpenCLError.h"
#include "../OpenCL/Include/OpenCLWorkFlow.h"
#include "../OpenCL/Include/RAII/OpenCLProgram.h"

void video_main(cl_device_id device, cl_context context);

#endif //NEW_YEAR_OPENCL_NEW_YEAR_CARD_VIDEO_H
