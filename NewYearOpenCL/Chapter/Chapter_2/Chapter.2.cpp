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


#include "../../OpenCL/Image/ImageChannelConvert.h"
#include "../../OpenCL/Image/ImageMask.h"
#include "../../OpenCL/Image/ImageCrop.h"
#include "../../OpenCL/Image/ImageResize.h"

#include "../../OpenCL/Include/OpenCLRAII.h"

#define ENABLE_CHAPTER_2_SECTION_1
#define ENABLE_CHAPTER_2_SECTION_2
#define ENABLE_CHAPTER_2_SECTION_3
#define ENABLE_CHAPTER_2_SECTION_4

extern float RatioVideoScale;
extern float RatioVideoFrame;

extern int CANVAS_WIDTH, CANVAS_HEIGHT;
extern int CANVAS_CENTER_X, CANVAS_CENTER_Y;
extern int FRAME_RATE;

const int CHAPTER_INDEX = 1;

cv::Mat chapter_2(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter* video_writer,
    cv::Mat* last_frame
) {
    std::cout << "Chapter 2" << std::endl;

    if (last_frame->channels() == 3) {
        cv::cvtColor(*last_frame, *last_frame, cv::COLOR_BGR2BGRA);
    }

    cv::Mat result_3channel(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC3);

    size_t global_work_size_2[2] = {static_cast<size_t>(CANVAS_WIDTH), static_cast<size_t>(CANVAS_HEIGHT)};

    const int frame_each_section = max_frame / 8;

    const auto queue = OpenCLQueue(context, device);
    // auto queue=CLCreateCommandQueue(context, device);

    auto program_channel = CLCreateProgram_Image_Channel(context, device);
    auto program_merge = CLCreateProgram_Image_Merge(context, device);
    auto program_resize = CLCreateProgram_Image_Resize(context, device);

    const int frame_section_1 = frame_each_section * 2;
#ifdef ENABLE_CHAPTER_2_SECTION_1

    const cv::Mat shmtu_logo = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

    const auto mem_shmtu_logo = OpenCLMem(
        context,
        shmtu_logo.cols, shmtu_logo.rows, shmtu_logo.channels(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        shmtu_logo.data
    );

    const auto mem_background = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        last_frame->data
    );

    const auto mem_frame_channel4 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto mem_frame_channel3 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 3
    );

    const auto frame_section_1_1 = static_cast<int>(frame_section_1 * 0.8);

    for (int i = 0; i < frame_section_1_1; ++i) {
        constexpr int logo_size = 400;
        const auto logo_new_size = static_cast<int>(
            RatioVideoScale *
            (
                1 + (logo_size - 1) *
                (
                    static_cast<float>(i) / static_cast<float>(frame_section_1_1 - 0)
                )
            )
        );

        const auto mem_shmtu_logo_resized = OpenCLMem(
            context,
            logo_new_size, logo_new_size, shmtu_logo.channels()
        );

        const auto kernel_resize = program_resize.CreateKernelRAII();
        KernelSetArg_Image_Resize(
            kernel_resize.GetKernel(),
            mem_shmtu_logo.GetMem(), mem_shmtu_logo_resized.GetMem(),
            shmtu_logo.cols, shmtu_logo.rows,
            logo_new_size, logo_new_size,
            shmtu_logo.channels()
        );
        size_t global_work_size_logo[2] = {static_cast<size_t>(logo_new_size), static_cast<size_t>(logo_new_size)};
        kernel_resize.KernelEnqueue(queue, 2, global_work_size_logo);

        const auto kernel_merge = program_merge.CreateKernelRAII();

        const auto alpha = static_cast<uchar>(
            255.0f *
            static_cast<float>(i) / static_cast<float>(frame_section_1_1 - 0)
        );
        KernelSetArg_Image_Merge(
            kernel_merge.GetKernel(),
            mem_background.GetMem(), mem_shmtu_logo_resized.GetMem(),
            mem_frame_channel4.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            CANVAS_CENTER_X - logo_new_size / 2, CANVAS_CENTER_Y - logo_new_size / 2,
            logo_new_size, logo_new_size, 4,
            alpha
        );
        kernel_merge.KernelEnqueue(queue, 2, global_work_size_2);

        const auto kernel_channel_convert = program_channel.CreateKernelRAII();
        KernelSetArg_Image_Channel(
            kernel_channel_convert.GetKernel(),
            mem_frame_channel4.GetMem(), mem_frame_channel3.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT,
            4, 3
        );
        kernel_channel_convert.KernelEnqueue(queue, 2, global_work_size_2);

        mem_frame_channel3.CopyToHost(queue.GetQueue(), result_3channel.data);

        // cv::imshow("result 3", result_3channel);
        // cv::waitKey(5);

        video_writer->write(result_3channel);
    }

    for (int i = frame_section_1; i < frame_section_1; ++i) {
        video_writer->write(result_3channel);
    }

#endif

#ifdef ENABLE_CHAPTER_2_SECTION_2

#endif

#ifdef ENABLE_CHAPTER_2_SECTION_3

#endif

#ifdef ENABLE_CHAPTER_2_SECTION_4

#endif

    // clReleaseCommandQueue(queue);

    last_frame->release();
    return result_3channel;
}
