// Chapter.3
// Created by Haomin Kong on 23-12-25.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "Chapter.2.h"

// 通道转换
#include "../../OpenCL/Image/ImageChannelConvert.h"

// 高斯模糊
#include "../../OpenCL/Image/ImageConvolution.h"
#include "../../OpenCL/Image/GaussianKernel.h"

// 图像合并
#include "../../OpenCL/Image/ImageMerge.h"

// 图像二值化
#include "../../OpenCL/Image/ImageBinarization.h"

// 图像掩膜
#include "../../OpenCL/Image/ImageMask.h"

// 图像裁剪
#include "../../OpenCL/Image/ImageCrop.h"

// 图像缩放
#include "../../OpenCL/Image/ImageResize.h"

// 图像填充
#include "../../OpenCL/Utils/OpenCLMemset.h"

// OpenCL RAII Model
#include "../../OpenCL/Include/OpenCLRAII.h"

// #define ENABLE_CHAPTER_3_SECTION_1
// #define ENABLE_CHAPTER_3_SECTION_2
#define ENABLE_CHAPTER_3_SECTION_3

#ifdef ENABLE_CHAPTER_2_SECTION_3
// Section 4 Must Based on Section 3
// #define ENABLE_CHAPTER_2_SECTION_4
#endif

extern float RatioVideoScale;
extern float RatioVideoFrame;

extern int CANVAS_WIDTH, CANVAS_HEIGHT;
extern int CANVAS_CENTER_X, CANVAS_CENTER_Y;
extern int FRAME_RATE;

const int CHAPTER_INDEX = 3;

cv::Mat chapter_3(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter* video_writer,
    cv::Mat* last_frame
) {
    std::cout << "Chapter 2" << std::endl;

    if (last_frame->channels() == 3) {
        cv::cvtColor(*last_frame, *last_frame, cv::COLOR_BGR2BGRA);
    }

    cv::Mat result_3channel(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC3);

    const auto mem_frame_channel3 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 3
    );

    size_t global_work_size_2[2] = {
        static_cast<size_t>(CANVAS_WIDTH),
        static_cast<size_t>(CANVAS_HEIGHT)
    };
    size_t global_work_size_3_4channel[3] = {
        static_cast<size_t>(CANVAS_WIDTH),
        static_cast<size_t>(CANVAS_HEIGHT),
        static_cast<size_t>(4)
    };

    const int frame_each_section = max_frame / 8;

    const auto queue = OpenCLQueue(context, device);
    // auto queue=CLCreateCommandQueue(context, device);

    auto program_channel = CLCreateProgram_Image_Channel(context, device);
    auto program_merge = CLCreateProgram_Image_Merge(context, device);
    auto program_resize = CLCreateProgram_Image_Resize(context, device);

    auto program_mask = CLCreateProgram_Image_Mask(context, device);
    auto program_crop = CLCreateProgram_Image_Crop(context, device);




    last_frame->release();
    return result_3channel;
}
