// Chapter.3
// Created by Haomin Kong on 23-12-25.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "Chapter.3.h"

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

#include "../../OpenCL/Image/Generate/GenerateGradientImage.h"

// 图像填充
#include "../../OpenCL/Utils/OpenCLMemset.h"

// OpenCL RAII Model
#include "../../OpenCL/Include/OpenCLRAII.h"

#define COLOR_COUNT 1024

#define ENABLE_CHAPTER_3_SECTION_1
#define ENABLE_CHAPTER_3_SECTION_2
// #define ENABLE_CHAPTER_3_SECTION_3

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
)
{
    std::cout << "Chapter 3" << std::endl;

    if (last_frame->channels() == 3)
    {
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

    auto program_gradient_color = CLCreateProgram_Generate_GradientColor(context, device);
    auto program_gradient_image = CLCreateProgram_Generate_GradientImage(context, device);

    const auto frame_section_1 = frame_each_section * 2;

    const auto max_r_center = std::sqrt(
        static_cast<float>(CANVAS_CENTER_X * CANVAS_CENTER_X + CANVAS_CENTER_Y * CANVAS_CENTER_Y)
    );
    const auto max_r = static_cast<float>(
        std::sqrt(
            static_cast<float>(CANVAS_WIDTH * CANVAS_WIDTH + CANVAS_HEIGHT * CANVAS_HEIGHT)
        )
    );

#ifdef ENABLE_CHAPTER_3_SECTION_1
    // Section 1
    std::cout << "Chapter " << CHAPTER_INDEX << " Section 1" << std::endl;

    const auto mem_last_frame = OpenCLMemFromHost(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4,
        last_frame->data
    );

    const auto mem_mask_output = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto mem_gradient_color = OpenCLMem(
        context,
        COLOR_COUNT, 1, 4
    );

    const auto mem_output_4channel = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    for (int frame_index = 0; frame_index < frame_section_1; ++frame_index)
    {
        const auto kernel_gradient_color = program_gradient_color.CreateKernelRAII();
        KernelSetArg_Generate_GradientColor(
            kernel_gradient_color,
            mem_gradient_color,
            COLOR_COUNT,
            0, 255, 255,
            0, 32, 161,
            4, 255
        );
        size_t global_work_size_color[1] = {
            static_cast<size_t>(COLOR_COUNT)
        };
        kernel_gradient_color.Execute(queue, 1, global_work_size_color);

        const auto mem_gradient_image = OpenCLMem(
            context,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4
        );
        const auto kernel_gradient_image = program_gradient_image.CreateKernelRAII();
        const int gradient_x = static_cast<int>(
            CANVAS_CENTER_X + max_r_center * std::cos(
                (
                    static_cast<float>(frame_index)
                    /
                    static_cast<float>(frame_each_section)
                )
                * 2 * M_PI
            )
        );
        const int gradient_y = static_cast<int>(
            CANVAS_CENTER_Y + max_r_center * std::sin(
                (
                    static_cast<float>(frame_index)
                    /
                    static_cast<float>(frame_each_section)
                )
                * 2 * M_PI
            )
        );
        KernelSetArg_Generate_GradientImage(
            kernel_gradient_image,
            mem_gradient_image,
            mem_gradient_color,
            COLOR_COUNT,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            gradient_x, gradient_y, max_r,
            4, 255
        );
        kernel_gradient_image.Execute(queue, 2, global_work_size_3_4channel);

        const auto kernel_mask = program_mask.CreateKernelRAII();
        const auto current_r = max_r_center *
        (
            1 - (
                static_cast<float>(frame_index)
                /
                static_cast<float>(frame_section_1)
            )
        );
        const auto light_source_y = static_cast<int>((-100) * RatioVideoScale);
        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            mem_last_frame, mem_mask_output,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            CANVAS_CENTER_X, CANVAS_CENTER_Y, current_r,
            0, 1,
            CANVAS_CENTER_X, light_source_y,
            0, 0, 0, 150
        );
        kernel_mask.Execute(queue, 2, global_work_size_2);

        const auto kernel_merge = program_merge.CreateKernelRAII();
        KernelSetArg_Image_Merge(
            kernel_merge,
            mem_gradient_image, mem_mask_output, mem_output_4channel,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            0, 0,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            255
        );
        kernel_merge.Execute(queue, 2, global_work_size_2);


        if (frame_index == frame_section_1 / 2)
        {
            // mem_last_frame.ShowByOpenCV(queue);
            // mem_mask_output.ShowByOpenCV(queue);
            // mem_output_4channel.ShowByOpenCV(queue);
        }


        // mem_gradient_image.ShowByOpenCV(queue);
        const auto kernel_channel = program_channel.CreateKernelRAII();
        KernelSetArg_Image_Channel(
            kernel_channel,
            mem_output_4channel, mem_frame_channel3,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            4, 3
        );
        kernel_channel.Execute(queue, 2, global_work_size_2);

        mem_frame_channel3.CopyToHost(queue, result_3channel.data);

        // cv::imshow("result", result_3channel);
        // cv::waitKey(15);
    }
#endif

#ifdef ENABLE_CHAPTER_3_SECTION_2

    const auto frame_section_2 = frame_each_section * 3;
    std::cout << "Chapter " << CHAPTER_INDEX << " Section 2" << std::endl;

    for (int frame_index = 0; frame_index < frame_section_2; ++frame_index)
    {
        const auto kernel_gradient_color = program_gradient_color.CreateKernelRAII();
        KernelSetArg_Generate_GradientColor(
            kernel_gradient_color,
            mem_gradient_color,
            COLOR_COUNT,
            0, 255, 255,
            0, 32, 161,
            4, 255
        );
        size_t global_work_size_color[1] = {
            static_cast<size_t>(COLOR_COUNT)
        };
        kernel_gradient_color.Execute(queue, 1, global_work_size_color);

        const auto mem_gradient_image = OpenCLMem(
            context,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4
        );
        const auto kernel_gradient_image = program_gradient_image.CreateKernelRAII();
        const int gradient_x = static_cast<int>(
            CANVAS_CENTER_X + max_r_center * std::cos(
                (
                    static_cast<float>(frame_index)
                    /
                    static_cast<float>(frame_each_section)
                )
                * 2 * M_PI
            )
        );
        const int gradient_y = static_cast<int>(
            CANVAS_CENTER_Y + max_r_center * std::sin(
                (
                    static_cast<float>(frame_index)
                    /
                    static_cast<float>(frame_each_section)
                )
                * 2 * M_PI
            )
        );
        KernelSetArg_Generate_GradientImage(
            kernel_gradient_image,
            mem_gradient_image,
            mem_gradient_color,
            COLOR_COUNT,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            gradient_x, gradient_y, max_r,
            4, 255
        );
        kernel_gradient_image.Execute(queue, 2, global_work_size_3_4channel);

        const auto kernel_mask = program_mask.CreateKernelRAII();
        const auto current_r = max_r_center *
        (

            static_cast<float>(frame_index)
            /
            static_cast<float>(frame_section_2)

        );
        const auto light_source_y = static_cast<int>((-100) * RatioVideoScale);
        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            mem_last_frame, mem_mask_output,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            CANVAS_CENTER_X, CANVAS_CENTER_Y, current_r,
            0, 1,
            CANVAS_CENTER_X, light_source_y,
            0, 0, 0, 150
        );
        kernel_mask.Execute(queue, 2, global_work_size_2);

        const auto kernel_merge = program_merge.CreateKernelRAII();
        KernelSetArg_Image_Merge(
            kernel_merge,
            mem_gradient_image, mem_mask_output, mem_output_4channel,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            0, 0,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            255
        );
        kernel_merge.Execute(queue, 2, global_work_size_2);


        if (frame_index == frame_section_1 / 2)
        {
            // mem_last_frame.ShowByOpenCV(queue);
            // mem_mask_output.ShowByOpenCV(queue);
            // mem_output_4channel.ShowByOpenCV(queue);
        }


        // mem_gradient_image.ShowByOpenCV(queue);
        const auto kernel_channel = program_channel.CreateKernelRAII();
        KernelSetArg_Image_Channel(
            kernel_channel,
            mem_output_4channel, mem_frame_channel3,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            4, 3
        );
        kernel_channel.Execute(queue, 2, global_work_size_2);

        mem_frame_channel3.CopyToHost(queue, result_3channel.data);
    }

#endif


    last_frame->release();
    return result_3channel;
}
