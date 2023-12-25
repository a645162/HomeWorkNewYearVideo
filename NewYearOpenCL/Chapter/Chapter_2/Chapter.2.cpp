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

#define ENABLE_CHAPTER_2_SECTION_1
#define ENABLE_CHAPTER_2_SECTION_2
#define ENABLE_CHAPTER_2_SECTION_3

#ifdef ENABLE_CHAPTER_2_SECTION_3
// Section 4 Must Based on Section 3
#define ENABLE_CHAPTER_2_SECTION_4
#endif

extern float RatioVideoScale;
extern float RatioVideoFrame;

extern int CANVAS_WIDTH, CANVAS_HEIGHT;
extern int CANVAS_CENTER_X, CANVAS_CENTER_Y;
extern int FRAME_RATE;

const int CHAPTER_INDEX = 2;

cv::Mat chapter_2(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter* video_writer,
    cv::Mat* last_frame
)
{
    std::cout << "Chapter 2" << std::endl;

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
    auto program_conv = CLCreateProgram_Image_Conv(context, device);
    auto program_mask = CLCreateProgram_Image_Mask(context, device);
    auto program_crop = CLCreateProgram_Image_Crop(context, device);
    auto program_binaryzation = CLCreateProgram_Image_Binaryzation(context, device);
    auto program_memset2d = CLCreateProgram_Memset_2D(context, device);

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

    const auto mem_background_blur = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto mem_frame_s1_channel4 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto frame_section_1_1 = static_cast<int>(frame_section_1 * 0.7);

    std::cout << "Chapter 2" << " Section 1" << std::endl;

    // Setcion 1 Main Loop
    for (int i = 0; i < frame_section_1_1; ++i) {
        const auto blur_intensity =
                100 * (
                    static_cast<float>(i)
                    /
                    static_cast<float>(frame_section_1_1)
                );
        // std::cout << "Blur Intensity: " << blur_intensity << std::endl;
        auto gaussian_params =
                calcGaussianKernelParameters(blur_intensity);
        auto kernel_gaussian = createGaussianKernel(gaussian_params);

        int gaussian_kernel_size = gaussian_params.size;
        const int conv_pad_size = gaussian_kernel_size / 2;

        const auto device_conv_kernel = OpenCLMem(
            context,
            gaussian_kernel_size * gaussian_kernel_size * sizeof(float),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (void *) kernel_gaussian
        );

        free(kernel_gaussian);

        const auto kernel_conv = program_conv.CreateKernelRAII();
        KernelSetArg_Image_Conv(
            kernel_conv,
            mem_background, mem_background_blur,
            CANVAS_HEIGHT, CANVAS_WIDTH, 4,
            device_conv_kernel,
            gaussian_kernel_size, conv_pad_size
        );
        kernel_conv.Execute(queue, 3, global_work_size_3_4channel);

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
        kernel_resize.Execute(queue, 2, global_work_size_logo);

        const auto kernel_merge = program_merge.CreateKernelRAII();

        const auto alpha = static_cast<uchar>(
            255.0f *
            static_cast<float>(i) / static_cast<float>(frame_section_1_1 - 0)
        );
        KernelSetArg_Image_Merge(
            kernel_merge.GetKernel(),
            mem_background_blur.GetMem(), mem_shmtu_logo_resized.GetMem(),
            mem_frame_s1_channel4.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            CANVAS_CENTER_X - logo_new_size / 2, CANVAS_CENTER_Y - logo_new_size / 2,
            logo_new_size, logo_new_size, 4,
            alpha
        );
        kernel_merge.Execute(queue, 2, global_work_size_2);

        const auto kernel_channel_convert = program_channel.CreateKernelRAII();
        KernelSetArg_Image_Channel(
            kernel_channel_convert.GetKernel(),
            mem_frame_s1_channel4.GetMem(), mem_frame_channel3.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT,
            4, 3
        );
        kernel_channel_convert.Execute(queue, 2, global_work_size_2);

        mem_frame_channel3.CopyToHost(queue.GetQueue(), result_3channel.data);

        video_writer->write(result_3channel);
    }

    for (int i = frame_section_1_1; i < frame_section_1; ++i) {
        video_writer->write(result_3channel);
    }

#endif

    const auto mem_frame_s2_channel4 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const int frame_section_2 = frame_each_section * 2;
#ifdef ENABLE_CHAPTER_2_SECTION_2
    std::cout << "Chapter 2" << " Section 2" << std::endl;
    const auto max_r = static_cast<float>(sqrt(
        CANVAS_CENTER_X * CANVAS_CENTER_X +
        CANVAS_CENTER_Y * CANVAS_CENTER_Y
    ));

    auto light_source_x = static_cast<int>(CANVAS_CENTER_X);
    auto light_source_y = static_cast<int>(-100 * RatioVideoScale);

    for (int i = 0; i < frame_section_2; ++i) {
        const auto radius =
                max_r - max_r * (
                    static_cast<float>(i)
                    /
                    static_cast<float>(frame_section_2)
                );

        const auto kernel_mask = program_mask.CreateKernelRAII();
        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            mem_frame_s1_channel4, mem_frame_s2_channel4,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            CANVAS_CENTER_X, CANVAS_CENTER_Y, radius,
            1, 1,
            light_source_x, light_source_y,
            0, 0, 0, 150
        );
        kernel_mask.Execute(queue, 2, global_work_size_2);

        // Channel Convert
        const auto kernel_channel_convert = program_channel.CreateKernelRAII();
        KernelSetArg_Image_Channel(
            kernel_channel_convert.GetKernel(),
            mem_frame_s2_channel4.GetMem(), mem_frame_channel3.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT,
            4, 3
        );
        kernel_channel_convert.Execute(queue, 2, global_work_size_2);
        mem_frame_channel3.CopyToHost(queue.GetQueue(), result_3channel.data);

        video_writer->write(result_3channel);
    }

#endif

    const int frame_section_3 = frame_each_section;
#ifdef ENABLE_CHAPTER_2_SECTION_3
    std::cout << "Chapter 2" << " Section 3" << std::endl;

    cv::Mat img2_origin =
        cv::imread("../Resources/Image/zhong_yuan_tu_shu_guan_.jpg");

    const int img2_origin_width = img2_origin.cols;
    const int img2_origin_height = img2_origin.rows;
    const int img2_origin_channels = img2_origin.channels();

    const auto mem_img2_origin = OpenCLMem(
        context,
        img2_origin_width, img2_origin_height, img2_origin_channels,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        img2_origin.data
    );

    unsigned int new_width = CANVAS_HEIGHT;
    unsigned int new_height = calculateNewHeightByNewWidth(
        img2_origin_width, img2_origin_height, new_width
    );
    if (new_height < static_cast<unsigned int>(CANVAS_HEIGHT))
    {
        new_height = CANVAS_HEIGHT;
        new_width = calculateNewWidthByNewHeight(
            img2_origin_width, img2_origin_height, new_height
        );
    }

    // std::cout << "new_width=" << new_width << " new_height=" << new_height << std::endl;

    const auto mem_img2_resized = OpenCLMem(
        context,
        new_width, new_height, img2_origin_channels
    );

    const auto kernel_resize = program_resize.CreateKernelRAII();
    KernelSetArg_Image_Resize(
        kernel_resize.GetKernel(),
        mem_img2_origin.GetMem(), mem_img2_resized.GetMem(),
        img2_origin_width, img2_origin_height,
        static_cast<int>(new_width),
        static_cast<int>(new_height),
        img2_origin_channels
    );
    // Because of here,global_work_size_2 is a const variable.
    // So The output may not full fill the Memory Area.
    // But we will crop them,so that it is doesn't matter.
    kernel_resize.Execute(queue, 2, global_work_size_2);

    const auto mem_img2_croped = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, img2_origin.channels()
    );
    const auto kernel_crop = program_crop.CreateKernelRAII();
    KernelSetArg_Image_Crop(
        kernel_crop.GetKernel(),
        mem_img2_resized.GetMem(), mem_img2_croped.GetMem(),
        static_cast<int>(new_width), static_cast<int>(new_height),
        CANVAS_WIDTH, CANVAS_HEIGHT,
        0, 0,
        CANVAS_WIDTH, CANVAS_HEIGHT,
        img2_origin.channels()
    );
    kernel_crop.Execute(queue, 2, global_work_size_2);

    const auto mem_img2_4channel = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto kernel_channel_img2 = program_channel.CreateKernelRAII();
    KernelSetArg_Image_Channel(
        kernel_channel_img2.GetKernel(),
        mem_img2_croped.GetMem(),
        mem_img2_4channel.GetMem(),
        CANVAS_WIDTH, CANVAS_HEIGHT,
        img2_origin.channels(), 4
    );
    kernel_channel_img2.Execute(queue, 2, global_work_size_2);

    // mem_img2_4channel.ShowByOpenCV(queue);

    constexpr float laplacian_conv_kernel[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    constexpr int laplacian_conv_kernel_size = 3;

    const auto mem_conv_kernel_laplacian = OpenCLMem(
        context,
        laplacian_conv_kernel_size * laplacian_conv_kernel_size * sizeof(float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (void*)(laplacian_conv_kernel)
    );

    const auto mem_img2_line_4channel = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto kernel_laplacian = program_conv.CreateKernelRAII();
    KernelSetArg_Image_Conv(
        kernel_laplacian.GetKernel(),
        mem_img2_4channel.GetMem(), mem_img2_line_4channel.GetMem(),
        CANVAS_HEIGHT, CANVAS_WIDTH, 4,
        mem_conv_kernel_laplacian,
        laplacian_conv_kernel_size,
        laplacian_conv_kernel_size / 2
    );
    kernel_laplacian.Execute(queue, 3, global_work_size_3_4channel);

    // mem_img2_line_4channel.ShowByOpenCV(queue);

    // cv::Mat img2_line_4channel(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC4);
    // mem_img2_line_4channel.CopyToHost(queue, img2_line_4channel.data);
    // cv::imwrite("img2_line_4channel.png",img2_line_4channel);

    const auto mem_img2_line_bin_4channel = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );
    const auto kernel_binaryzation = program_binaryzation.CreateKernelRAII();
    KernelSetArg_Image_Binaryzation(
        kernel_binaryzation.GetKernel(),
        mem_img2_line_4channel.GetMem(),
        mem_img2_line_bin_4channel.GetMem(),
        CANVAS_WIDTH, CANVAS_HEIGHT, 4,
        30, false
    );
    kernel_binaryzation.Execute(queue, WORK_DIM_IMAGE_BINARYZATION, global_work_size_2);

    // mem_img2_line_bin_4channel.ShowByOpenCV(queue);

    // Frame Section 3(Channel 4)
    const auto mem_frame_s3_channel4 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    const auto mem_background_black = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    // Apply Background to Black Color
    const auto kernel_memset = program_memset2d.CreateKernelRAII();
    KernelSetArg_Memset_2D(
        kernel_memset.GetKernel(),
        mem_background_black.GetMem(),
        CANVAS_WIDTH, CANVAS_HEIGHT, 4,
        0
    );
    kernel_memset.Execute(queue, 2, global_work_size_2);

    for (int i = 0; i < frame_section_3; ++i)
    {
        const auto alpha = static_cast<uchar>(
            255.0f *
            (static_cast<float>(i) / static_cast<float>(frame_section_3 - 0))
        );

        const auto kernel_merge_line = program_merge.CreateKernelRAII();
        KernelSetArg_Image_Merge(
            kernel_merge_line.GetKernel(),
            mem_background_black.GetMem(),
            mem_img2_line_bin_4channel.GetMem(),
            mem_frame_s3_channel4.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            0, 0,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            alpha
        );
        kernel_merge_line.Execute(queue, 2, global_work_size_2);

        cv::Mat result_4channel(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC4);
        mem_frame_s3_channel4.CopyToHost(queue.GetQueue(), result_4channel.data);
        cv::cvtColor(result_4channel, result_3channel, cv::COLOR_BGRA2BGR);
        video_writer->write(result_3channel);

        // if (i == frame_section_3 - 1) {
        //     std::cout << "Last Frame on Section 3" << std::endl;
        //     mem_frame_s3_channel4.ShowByOpenCV(queue);
        // }

        // // TODO:Channel Convert
        // const auto kernel_channel_convert1 = program_channel.CreateKernelRAII();
        // KernelSetArg_Image_Channel(
        //     kernel_channel_convert1.GetKernel(),
        //     mem_frame_s3_channel4.GetMem(),
        //     mem_frame_channel3.GetMem(),
        //     CANVAS_WIDTH, CANVAS_HEIGHT,
        //     4, 3
        // );
        // kernel_channel_convert1.Execute(queue, 2, global_work_size_2);
        //
        // if (i == frame_section_3 - 1) {
        //     std::cout << "Last Frame on Section 3" << std::endl;
        //     mem_frame_channel3.ShowByOpenCV(queue);
        // }
        //
        // mem_frame_channel3.CopyToHost(queue.GetQueue(), result_3channel.data);
        // video_writer->write(result_3channel);
    }


#endif

    const int frame_section_4 = max_frame - frame_section_1 - frame_section_2 - frame_section_3;
#ifdef ENABLE_CHAPTER_2_SECTION_4

    std::cout << "Chapter 2" << " Section 4" << std::endl;

    // Based on Section 3 Result
    // const auto mem_section4_background = OpenCLMem(
    //     context,
    //     CANVAS_WIDTH, CANVAS_HEIGHT, 4
    // );
    // mem_section4_background.CopyFromOtherMem(
    //     queue,
    //     mem_frame_s3_channel4
    // );

    // Frame Section 4(Channel 4)
    const auto mem_frame_s4_channel4 = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, 4
    );

    // Cover mem_img2_4channel to mem_frame_s3_channel4
    for (int i = 0; i < frame_section_4; ++i)
    {
        const auto alpha = static_cast<uchar>(
            255.0f *
            (static_cast<float>(i) / static_cast<float>(frame_section_4 - 0))
        );

        // Merge
        const auto kernel_merge = program_merge.CreateKernelRAII();
        KernelSetArg_Image_Merge(
            kernel_merge.GetKernel(),
            mem_frame_s3_channel4.GetMem(),
            mem_img2_4channel.GetMem(),
            mem_frame_s4_channel4.GetMem(),
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            0, 0,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            alpha
        );
        kernel_merge.Execute(queue, 2, global_work_size_2);

        cv::Mat result_4channel(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC4);
        mem_frame_s4_channel4.CopyToHost(queue.GetQueue(), result_4channel.data);
        cv::cvtColor(result_4channel, result_3channel, cv::COLOR_BGRA2BGR);
        video_writer->write(result_3channel);

        // cv::imshow("Section 4", result_3channel);
        // cv::waitKey(15);

        // // Channel Convert
        // const auto kernel_channel_convert = program_channel.CreateKernelRAII();
        // KernelSetArg_Image_Channel(
        //     kernel_channel_convert.GetKernel(),
        //     mem_frame_s4_channel4.GetMem(),
        //     mem_frame_channel3.GetMem(),
        //     CANVAS_WIDTH, CANVAS_HEIGHT,
        //     4, 3
        // );
        // kernel_channel_convert.Execute(queue, 2, global_work_size_2);
        //
        // mem_frame_channel3.CopyToHost(queue.GetQueue(), result_3channel.data);
        // video_writer->write(result_3channel);
    }

#endif

    last_frame->release();
    return result_3channel;
}
