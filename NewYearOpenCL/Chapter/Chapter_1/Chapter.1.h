// Chapter.1
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_VIDEO_CHAPTER_1_H
#define NEW_YEAR_OPENCL_VIDEO_CHAPTER_1_H

#include "../../OpenCV/Include/OpenCVInclude.h"

#include "../../OpenCL/Include/OpenCLInclude.h"
#include "../../OpenCL/Include/OpenCLError.h"
#include "../../OpenCL/Include/OpenCLFlow.h"
#include "../../OpenCL/Include/OpenCLProgram.h"

cv::Mat chapter_1(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter* video_writer
);

#endif //NEW_YEAR_OPENCL_VIDEO_CHAPTER_1_H
