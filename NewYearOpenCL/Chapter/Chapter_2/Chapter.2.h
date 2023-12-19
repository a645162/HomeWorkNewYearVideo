// Chapter.2
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_VIDEO_CHAPTER_2_H
#define NEW_YEAR_OPENCL_VIDEO_CHAPTER_2_H

#include "../../OpenCV/Include/OpenCVInclude.h"

#include "../../OpenCL/Include/OpenCLInclude.h"
#include "../../OpenCL/Include/OpenCLError.h"
#include "../../OpenCL/Include/OpenCLFlow.h"
#include "../../OpenCL/Include/OpenCLProgram.h"

cv::Mat chapter_2(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter *video_writer,
    cv::Mat *last_frame
);

#endif //NEW_YEAR_OPENCL_VIDEO_CHAPTER_2_H
