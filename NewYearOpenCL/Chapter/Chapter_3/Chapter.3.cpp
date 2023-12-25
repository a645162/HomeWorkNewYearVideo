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

#include "../../OpenCL/Image/Draw/DrawRect.h"

#include "../../OpenCL/Image/ImageReverseColor.h"

// 图像填充
#include "../../OpenCL/Utils/OpenCLMemset.h"

// OpenCL RAII Model
#include "../../OpenCL/Include/OpenCLRAII.h"

#define COLOR_COUNT 1024

#define ENABLE_CHAPTER_3_SECTION_1
#define ENABLE_CHAPTER_3_SECTION_2
#define ENABLE_CHAPTER_3_SECTION_3


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

    auto program_draw_rect = CLCreateProgram_Draw_Rect(context, device);

    auto program_reverse_color = CLCreateProgram_Image_Reverse_Color(context, device);

    auto program_memset_2d = CLCreateProgram_Memset_2D(context, device);

    const auto frame_section_1 = frame_each_section * 2;

    const auto max_r_center = std::sqrt(
        static_cast<float>(CANVAS_CENTER_X * CANVAS_CENTER_X + CANVAS_CENTER_Y * CANVAS_CENTER_Y)
    );
    const auto max_r = static_cast<float>(
        std::sqrt(
            static_cast<float>(CANVAS_WIDTH * CANVAS_WIDTH + CANVAS_HEIGHT * CANVAS_HEIGHT)
        )
    );

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

#ifdef ENABLE_CHAPTER_3_SECTION_1
    // Section 1
    std::cout << "Chapter " << CHAPTER_INDEX << " Section 1" << std::endl;

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
        video_writer->write(result_3channel);
    }
#endif

    const auto frame_section_2 = frame_each_section * 1;

    const int x1 = static_cast<int>(100 * RatioVideoScale), y1 = static_cast<int>(100 * RatioVideoScale);
    const int x2 = CANVAS_WIDTH - x1, y2 = CANVAS_HEIGHT - y1;
    const int thickness = static_cast<int>(10 * RatioVideoScale);

    const int rect_width = static_cast<int>(x2 - x1);
    const int rect_height = static_cast<int>(y2 - y1);
    const float rect_max_r = std::sqrt(
        static_cast<float>(rect_width * rect_width + rect_height * rect_height)
    ) / 2;

    const int inner_start_x = static_cast<int>(x1 + thickness);
    const int inner_start_y = static_cast<int>(y1 + thickness);
    const int inner_width = static_cast<int>(x2 - x1 - thickness * 2);
    const int inner_height = static_cast<int>(y2 - y1 - thickness * 2);

    const auto mem_rect_output = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

#ifdef ENABLE_CHAPTER_3_SECTION_2

    std::cout << "Chapter " << CHAPTER_INDEX << " Section 2" << std::endl;

    {
        const auto kernel_memset_2d = program_memset_2d.CreateKernelRAII();
        KernelSetArg_Memset_2D(
            kernel_memset_2d,
            mem_rect_output,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            0
        );
        kernel_memset_2d.Execute(queue, 2, global_work_size_2);
    }

    {
        const auto kernel_draw_rect = program_draw_rect.CreateKernelRAII();
        KernelSetArg_Draw_Rect(
            kernel_draw_rect,
            mem_rect_output,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            x1, y1,
            x2, y2,
            thickness,
            255, 255, 255,
            238, 238, 238,
            4,
            true,
            true, 0.01f
        );
        kernel_draw_rect.Execute(queue, 2, global_work_size_2);
        // mem_rect_output.ShowByOpenCV(queue);
    }

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
        const auto current_r = rect_max_r *
        (
            static_cast<float>(frame_index)
            /
            static_cast<float>(frame_section_2)
        );
        // const auto light_source_y = static_cast<int>((-100) * RatioVideoScale);
        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            mem_rect_output, mem_mask_output,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            CANVAS_CENTER_X, CANVAS_CENTER_Y, current_r,
            0, 0,
            0, 0,
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
        // cv::waitKey(10);
        video_writer->write(result_3channel);
    }

#endif

    const auto frame_section_3 = max_frame - frame_section_1 - frame_section_2;
#ifdef ENABLE_CHAPTER_3_SECTION_3

    std::cout << "Chapter " << CHAPTER_INDEX << " Section 3" << std::endl;

    auto host_image_text =
        cv::imread("../Resources/Image/NewYearText.png", cv::IMREAD_UNCHANGED);
    if (host_image_text.channels() == 3)
    {
        cv::cvtColor(host_image_text, host_image_text, cv::COLOR_BGR2RGBA);
    }

    const auto ori_width = host_image_text.cols;
    const auto ori_height = host_image_text.rows;

    auto new_width = inner_width;
    auto new_height = static_cast<int>(calculateNewHeightByNewWidth(
        ori_width, ori_height,
        new_width
    ));

    if (new_height > inner_height)
    {
        new_height = inner_height;
        new_width = static_cast<int>(calculateNewWidthByNewHeight(
            ori_width, ori_height,
            new_height
        ));
    }

    int target_start_x = inner_start_x + (inner_width - new_width) / 2;
    int target_start_y = inner_start_y + (inner_height - new_height) / 2;

    const auto mem_text_ori = OpenCLMemFromHost(
        context,
        ori_width, ori_height, 4,
        host_image_text.data
    );

    const auto mem_text_output = OpenCLMem(
        context,
        new_width, new_height, 4
    );

    const auto mem_text_reverse = OpenCLMem(
        context,
        new_width, new_height, 4
    );

    {
        const auto kernel_resize = program_resize.CreateKernelRAII();
        KernelSetArg_Image_Resize(
            kernel_resize,
            mem_text_ori, mem_text_output,
            ori_width, ori_height,
            new_width, new_height,
            4
        );
        size_t text_global_work_size[2] = {
            static_cast<size_t>(new_width),
            static_cast<size_t>(new_height)
        };
        kernel_resize.Execute(queue, 2, text_global_work_size);
        // mem_text_output.ShowByOpenCV(queue);
    }

    {
        const auto kernel_reverse = program_reverse_color.CreateKernelRAII();
        KernelSetArg_Image_Reverse_Color(
            kernel_reverse,
            mem_text_output, mem_text_reverse,
            new_width, new_height,
            4
        );
        size_t text_global_work_size[2] = {
            static_cast<size_t>(new_width),
            static_cast<size_t>(new_height)
        };
        kernel_reverse.Execute(queue, 2, text_global_work_size);
        // mem_text_reverse.ShowByOpenCV(queue);
    }

    const int reverse_frame = frame_section_3 / 10;
    bool is_reverse = true;

    const uchar max_alpha = 255, min_alpha = 50;
    const float max_freq = 1.0f, min_freq = 0.01f;

    const auto mem_merged_notext = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    for (int frame_index = 0; frame_index < frame_section_3; ++frame_index)
    {
        if (frame_index % reverse_frame == 0)
        {
            is_reverse = !is_reverse;
        }

        auto current_mem_text = &mem_text_output;
        if (is_reverse)
        {
            current_mem_text = &mem_text_reverse;
        }

        auto alpha_rate = static_cast<float>(frame_index % reverse_frame) / static_cast<float>(reverse_frame);
        if (is_reverse)
        {
            alpha_rate = 1 - alpha_rate;
        }
        const auto text_alpha = static_cast<uchar>(min_alpha + alpha_rate * (max_alpha - min_alpha));

        const auto current_rect_freq = static_cast<float>(min_freq + alpha_rate * (max_freq - min_freq));

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

        {
            const auto kernel_draw_rect = program_draw_rect.CreateKernelRAII();
            KernelSetArg_Draw_Rect(
                kernel_draw_rect,
                mem_rect_output,
                CANVAS_WIDTH, CANVAS_HEIGHT,
                x1, y1,
                x2, y2,
                thickness,
                255, 255, 255,
                238, 238, 238,
                4,
                true,
                true, current_rect_freq
            );
            kernel_draw_rect.Execute(queue, 2, global_work_size_2);
            // mem_rect_output.ShowByOpenCV(queue);
        }

        const auto kernel_merge = program_merge.CreateKernelRAII();
        KernelSetArg_Image_Merge(
            kernel_merge,
            mem_gradient_image, mem_rect_output, mem_merged_notext,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            0, 0,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            255
        );
        kernel_merge.Execute(queue, 2, global_work_size_2);

        KernelSetArg_Image_Merge(
            kernel_merge,
            mem_merged_notext, *current_mem_text, mem_output_4channel,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            target_start_x, target_start_y,
            new_width, new_height, 4,
            text_alpha
        );
        kernel_merge.Execute(queue, 2, global_work_size_2);

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
        // cv::waitKey(10);
        video_writer->write(result_3channel);
    }


#endif

    last_frame->release();
    return result_3channel;
}
