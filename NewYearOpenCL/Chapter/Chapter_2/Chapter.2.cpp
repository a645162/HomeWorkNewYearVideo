// Chapter.2
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "Chapter.2.h"

// 通道转换
#include "../../OpenCL/Image/ImageChannelConvert.h"

// 高斯模糊
#include "../../OpenCL/Image/ImageConvolution.h"
#include "../../OpenCL/Image/GaussianKernel.h"

// 图像合并
#include "../../OpenCL/Image/ImageMerge.h"


#include "../../OpenCL/Image/ImageMerge.h"
#include "../../OpenCL/Image/ImageMask.h"
#include "../../OpenCL/Image/ImageCrop.h"

#define ENABLE_CHAPTER_2_SECTION_1
#define ENABLE_CHAPTER_2_SECTION_2
#define ENABLE_CHAPTER_2_SECTION_3
#define ENABLE_CHAPTER_2_SECTION_4

cv::Mat chapter_2(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter *video_writer,
    cv::Mat *last_frame
) {
    cv::Mat result;


#ifdef ENABLE_CHAPTER_2_SECTION_1

#endif



    return result;
}
