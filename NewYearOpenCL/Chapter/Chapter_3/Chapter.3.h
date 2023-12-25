// Chapter.3
// Created by Haomin Kong on 23-12-25.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_VIDEO_CHAPTER_3_H
#define NEW_YEAR_OPENCL_VIDEO_CHAPTER_3_H

#include "../../OpenCV/Include/OpenCVInclude.h"

#include "../../OpenCL/Include/OpenCLInclude.h"
#include "../../OpenCL/Include/OpenCLError.h"
#include "../../OpenCL/Include/OpenCLWorkFlow.h"
#include "../../OpenCL/Include/RAII/OpenCLProgram.h"

cv::Mat chapter_3(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter* video_writer,
    cv::Mat* last_frame
);

#endif //NEW_YEAR_OPENCL_VIDEO_CHAPTER_3_H
