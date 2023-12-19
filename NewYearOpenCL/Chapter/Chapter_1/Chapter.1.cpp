// Chapter.1
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "Chapter.1.h"

#include "../../Utils/ProgramLog.h"

// 通道转换
#include "../../OpenCL/Image/ImageChannelConvert.h"

// 图像缩放
#include "../../OpenCL/Image/ImageResize.h"

// 图像合并
#include "../../OpenCL/Image/ImageMerge.h"

// 图像遮罩
#include "../../OpenCL/Image/ImageMask.h"

// 图像裁剪
#include "../../OpenCL/Image/ImageCrop.h"

// OpenCL Memset
#include "../../OpenCL/Utils/OpenCLMemset.h"

#define ENABLE_CHAPTER_1_SECTION_1
#define ENABLE_CHAPTER_1_SECTION_2
#define ENABLE_CHAPTER_1_SECTION_3
#define ENABLE_CHAPTER_1_SECTION_4

extern float RatioVideoScale;
extern float RatioVideoFrame;

extern int CANVAS_WIDTH, CANVAS_HEIGHT;
extern int CANVAS_CENTER_X, CANVAS_CENTER_Y;
extern int FRAME_RATE;

constexpr auto CANVAS_CHANNEL = 4;

const int chapter_index = 1;

cv::Mat chapter_1(
    cl_context context, cl_device_id device,
    int max_frame, cv::VideoWriter* video_writer
) {
    max_frame = static_cast<int>(static_cast<float>(max_frame) * RatioVideoFrame);

    std::cout << "Chapter 1" << std::endl;
    const int frame_pre_section = max_frame / 7;

    // Section 1

    auto logo_start_size = static_cast<int>(std::max(CANVAS_WIDTH, CANVAS_HEIGHT) * 1.2);
    auto logo_min_size = static_cast<int>(400 * RatioVideoScale);

    // Read input image
    cv::Mat shmtu_logo = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

    // Create White Canvas
    // cv::Mat canvas = cv::Mat::zeros(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC4);
    // canvas.setTo(cv::Scalar(255, 255, 255, 255));

    int logo_src_width = shmtu_logo.cols;
    int logo_src_height = shmtu_logo.rows;
    int logo_src_channels = shmtu_logo.channels();

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_resize = CLCreateProgram_Image_Resize(context, device);
    OpenCLProgram program_merge = CLCreateProgram_Image_Merge(context, device);
    OpenCLProgram program_channel = CLCreateProgram_Image_Channel(context, device);
    OpenCLProgram program_crop = CLCreateProgram_Image_Crop(context, device);
    OpenCLProgram program_mask = CLCreateProgram_Image_Mask(context, device);
    OpenCLProgram program_memset = CLCreateProgram_Memset_2D(context, device);

    cv::Mat result(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC(3));

#ifdef ENABLE_CHAPTER_1_SECTION_1
    // cl_mem device_canvas_ori = OpenCLMalloc(
    //     context,
    //     CANVAS_WIDTH * CANVAS_HEIGHT * logo_src_channels * sizeof(uchar),
    //     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    //     canvas.data
    // );


    auto device_canvas_ori = OpenCLMem(
        context,
        CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_CHANNEL
    );

    auto kernel_memset = program_memset.CreateKernelRAII();
    KernelSetArg_Memset_2D(
        kernel_memset.GetKernel(),
        device_canvas_ori.GetMem(),
        CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_CHANNEL,
        255
    );
    size_t global_work_size_memset[3] = {
        static_cast<size_t>(CANVAS_WIDTH),
        static_cast<size_t>(CANVAS_HEIGHT),
        static_cast<size_t>(CANVAS_CHANNEL)
    };
    CLKernelEnqueue(
        queue, kernel_memset.GetKernel(),
        3, global_work_size_memset
    );
    clFinish(queue);

    cl_mem device_logo_ori = OpenCLMalloc(
        context,
        logo_src_width * logo_src_height * logo_src_channels * sizeof(uchar),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        shmtu_logo.data
    );

    cl_mem device_merge_target = OpenCLMalloc(
        context,
        CANVAS_WIDTH * CANVAS_HEIGHT * logo_src_channels * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );

    cl_mem device_output_3channel = OpenCLMalloc(
        context,
        CANVAS_WIDTH * CANVAS_HEIGHT * 3 * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );

    const auto section_1_frame = frame_pre_section * 2;
    const auto section_1_frame_effect = static_cast<int>(section_1_frame * 0.8);
    std::cout << "Chapter 1 Section 1" << std::endl;
    for (int i = 0; i < section_1_frame_effect; ++i) {
        //        output_frame_log(chapter_index, 1, i, section_1_frame);
        auto current_size =
                logo_start_size -
                static_cast<int>(
                    (
                        static_cast<float>(i) / static_cast<float>(section_1_frame_effect - 0)
                    )
                    *
                    static_cast<float>(logo_start_size - logo_min_size)
                );


        const int merge_target_x = CANVAS_CENTER_X - current_size / 2;
        const int merge_target_y = CANVAS_CENTER_Y - current_size / 2;

        cl_mem device_resized_logo = OpenCLMalloc(
            context,
            current_size * current_size * logo_src_channels * sizeof(uchar),
            CL_MEM_READ_WRITE,
            nullptr
        );

        cl_kernel kernel_resize_1 = program_resize.CreateKernel();

        KernelSetArg_Image_Resize(
            kernel_resize_1,
            device_logo_ori, device_resized_logo,
            logo_src_width, logo_src_height,
            current_size, current_size,
            logo_src_channels
        );

        size_t logo_global_work_size[2] =
                {static_cast<size_t>(current_size), static_cast<size_t>(current_size)};

        // Execute the OpenCL kernel
        CLKernelEnqueue(
            queue, kernel_resize_1,
            2, logo_global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_resize_1);

        cl_kernel kernel_merge = program_merge.CreateKernel();
        size_t global_work_size[2] = {static_cast<size_t>(CANVAS_WIDTH), static_cast<size_t>(CANVAS_HEIGHT)};
        KernelSetArg_Image_Merge(
            kernel_merge,
            device_canvas_ori.GetMem(), device_resized_logo, device_merge_target,
            CANVAS_WIDTH, CANVAS_HEIGHT, logo_src_channels,
            merge_target_x, merge_target_y,
            current_size, current_size, logo_src_channels,
            255
        );

        CLKernelEnqueue(
            queue, kernel_merge,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseMemObject(device_resized_logo);
        clReleaseKernel(kernel_merge);

        cl_kernel kernel_channel = program_channel.CreateKernel();

        KernelSetArg_Image_Channel(
            kernel_channel,
            device_merge_target, device_output_3channel,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            logo_src_channels, 3
        );
        CLKernelEnqueue(
            queue, kernel_channel,
            2, global_work_size
        );

        clFinish(queue);
        clReleaseKernel(kernel_channel);

        OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_output_3channel,
            CANVAS_WIDTH * CANVAS_HEIGHT * 3 * sizeof(uchar)
        );

        video_writer->write(result);
    }

    for (int i = section_1_frame_effect; i < section_1_frame; ++i) {
        //        output_frame_log(chapter_index, 1, i, section_1_frame);
        video_writer->write(result);
    }

    // clReleaseMemObject(device_canvas_ori);
    clReleaseMemObject(device_logo_ori);
    clReleaseMemObject(device_merge_target);
    clReleaseMemObject(device_output_3channel);

    program_merge.ReleaseProgram();
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Section 2
    const auto section_2_frame = frame_pre_section * 2;
    cv::Mat img_school_door = cv::imread("../Resources/Image/IMG_3493.jpg");
    const auto img_school_door_width_ori = img_school_door.cols;
    const auto img_school_door_height_ori = img_school_door.rows;
    const auto img_school_door_channels = img_school_door.channels();

    auto img_school_door_height = CANVAS_HEIGHT;
    auto img_school_door_width = static_cast<int>(
        calculateNewWidthByNewHeight(
            img_school_door_width_ori, img_school_door_height_ori,
            img_school_door_height
        )
    );

    // When Width is not enough
    if (img_school_door_width < CANVAS_WIDTH) {
        img_school_door_width = CANVAS_WIDTH;
        img_school_door_height = static_cast<int>(
            calculateNewHeightByNewWidth(
                img_school_door_width_ori, img_school_door_height_ori,
                img_school_door_width
            )
        );
    }

    cl_mem device_img_school_door_ori = OpenCLMalloc(
        context,
        img_school_door_width_ori * img_school_door_height_ori * img_school_door_channels * sizeof(uchar),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        img_school_door.data
    );
    // Resize Ori Image
    cl_mem device_img_school_door_resized_ori_channel = OpenCLMalloc(
        context,
        img_school_door_width * img_school_door_height * img_school_door_channels * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );
    auto kernel_resize = program_resize.CreateKernel();
    KernelSetArg_Image_Resize(
        kernel_resize,
        device_img_school_door_ori, device_img_school_door_resized_ori_channel,
        img_school_door_width_ori, img_school_door_height_ori,
        img_school_door_width, img_school_door_height,
        img_school_door_channels
    );
    size_t global_work_size_ori[2] = {
        static_cast<size_t>(img_school_door_width),
        static_cast<size_t>(img_school_door_height)
    };
    CLKernelEnqueue(
        queue, kernel_resize,
        2, global_work_size_ori
    );
    clFinish(queue);
    clReleaseKernel(kernel_resize);
    clReleaseMemObject(device_img_school_door_ori);
    program_resize.ReleaseProgram();

    // Crop
    auto x_1 = img_school_door_width / 2 - CANVAS_WIDTH / 2;
    auto x_2 = img_school_door_width / 2 + CANVAS_WIDTH / 2;
    auto y_1 = 0;
    auto y_2 = img_school_door_height;

    cl_mem device_img_school_door_resized_ori_channel_cropped = OpenCLMalloc(
        context,
        CANVAS_WIDTH * CANVAS_HEIGHT * img_school_door_channels * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );

    size_t global_work_size[2] = {
        static_cast<size_t>(CANVAS_WIDTH),
        static_cast<size_t>(CANVAS_HEIGHT)
    };

    auto kernel_crop = program_crop.CreateKernel();
    KernelSetArg_Image_Crop(
        kernel_crop,
        device_img_school_door_resized_ori_channel,
        device_img_school_door_resized_ori_channel_cropped,
        img_school_door_width, img_school_door_height,
        CANVAS_WIDTH, CANVAS_HEIGHT,
        x_1, y_1, x_2, y_2,
        img_school_door_channels
    );
    CLKernelEnqueue(
        queue, kernel_crop,
        2, global_work_size
    );
    clFinish(queue);
    clReleaseKernel(kernel_crop);
    clReleaseMemObject(device_img_school_door_resized_ori_channel);
    program_crop.ReleaseProgram();

    img_school_door_width = CANVAS_WIDTH;
    img_school_door_height = CANVAS_HEIGHT;

    // Convert to 4 Channel
    cl_mem device_img_school_door_4channel = OpenCLMalloc(
        context,
        img_school_door_width * img_school_door_height * 4 * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );

    cl_kernel kernel_channel = program_channel.CreateKernel();
    KernelSetArg_Image_Channel(
        kernel_channel,
        device_img_school_door_resized_ori_channel_cropped,
        device_img_school_door_4channel,
        img_school_door_width, img_school_door_height,
        img_school_door_channels, 4
    );

    CLKernelEnqueue(
        queue, kernel_channel,
        2, global_work_size
    );
    clFinish(queue);
    clReleaseKernel(kernel_channel);
    clReleaseMemObject(device_img_school_door_resized_ori_channel_cropped);

    // Mask
    cl_mem device_img_school_door_mask_output = OpenCLMalloc(
        context,
        CANVAS_WIDTH * CANVAS_HEIGHT * 4 * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );
    cl_mem device_img_school_door_frame_output = OpenCLMalloc(
        context,
        CANVAS_WIDTH * CANVAS_HEIGHT * 3 * sizeof(uchar),
        CL_MEM_READ_WRITE,
        nullptr
    );
    //    program_crop.ReleaseProgram();
    //    OpenCLProgram program_mask = CLCreateProgram_Image_Mask(context, device);
#ifdef ENABLE_CHAPTER_1_SECTION_2
    std::cout << "Chapter 1 Section 2" << std::endl;
    for (int i = 0; i < section_2_frame; ++i) {
        //        output_frame_log(chapter_index, 2, i, section_2_frame);

        auto radius = 150 * RatioVideoScale;

        auto start_x = -radius * 2 + static_cast<int>(
                           static_cast<float>(i) / static_cast<float>(section_2_frame) * (CANVAS_WIDTH + radius * 2)
                       );

        cl_kernel kernel_mask = program_mask.CreateKernel();

        auto center_x = static_cast<int>(start_x + radius);
        auto center_y = static_cast<int>(
            CANVAS_HEIGHT
            *
            (0.8 - (static_cast<float>(i) / static_cast<float>(section_2_frame)) * 0.3)
        );

        auto light_source_x = static_cast<int>(
            CANVAS_WIDTH -
            static_cast<float>(img_school_door_width / 2) * static_cast<float>(i) /
            static_cast<float>(section_2_frame)
        );
        auto light_source_y = static_cast<int>(-100 * RatioVideoScale);

        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            device_img_school_door_4channel, device_img_school_door_mask_output,
            CANVAS_WIDTH, CANVAS_HEIGHT, 4,
            center_x, center_y, radius,
            1, 1,
            light_source_x, light_source_y,
            0, 0, 0, 150
        );
        CLKernelEnqueue(
            queue, kernel_mask,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_mask);

        // Channel Convert
        auto kernel_channel_1 = program_channel.CreateKernel();
        KernelSetArg_Image_Channel(
            kernel_channel_1,
            device_img_school_door_mask_output,
            device_img_school_door_frame_output,
            CANVAS_WIDTH, CANVAS_HEIGHT,
            4, 3
        );
        CLKernelEnqueue(
            queue, kernel_channel_1,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_channel_1);

        OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_img_school_door_frame_output,
            CANVAS_WIDTH * CANVAS_HEIGHT * 3 * sizeof(uchar)
        );

        //        cv::imshow("result", result);
        //        cv::waitKey(10);

        video_writer->write(result);
    }
#endif

    const auto section_3_frame = frame_pre_section * 1;
#ifdef ENABLE_CHAPTER_1_SECTION_3
    std::cout << "Chapter 1 Section 3" << std::endl;
    for (int i = 0; i < section_3_frame; ++i) {
        //        output_frame_log(chapter_index, 3, i, section_3_frame);

        auto radius = 150 * RatioVideoScale;

        cl_kernel kernel_mask = program_mask.CreateKernel();

        auto center_x = static_cast<int>(static_cast<float>(CANVAS_WIDTH) + radius)
                        - static_cast<int>(
                            (static_cast<float>(CANVAS_WIDTH / 2) + radius)
                            * static_cast<float>(i) / static_cast<float>(section_3_frame)
                        );
        auto center_y = static_cast<int>(
            CANVAS_HEIGHT
            *
            (0.5)
        );

        auto light_source_x = (img_school_door_width / 2);
        auto light_source_y = static_cast<int>(-100 * RatioVideoScale);

        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            device_img_school_door_4channel, device_img_school_door_mask_output,
            img_school_door_width, img_school_door_height, 4,
            center_x, center_y, radius,
            1, 1,
            light_source_x, light_source_y,
            0, 0, 0, 150
        );
        CLKernelEnqueue(
            queue, kernel_mask,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_mask);

        // Channel Convert
        auto kernel_channel_1 = program_channel.CreateKernel();
        KernelSetArg_Image_Channel(
            kernel_channel_1,
            device_img_school_door_mask_output,
            device_img_school_door_frame_output,
            img_school_door_width, img_school_door_height,
            4, 3
        );
        CLKernelEnqueue(
            queue, kernel_channel_1,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_channel_1);

        OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_img_school_door_frame_output,
            img_school_door_width * img_school_door_height * 3 * sizeof(uchar)
        );

        //        cv::imshow("result", result);
        //        cv::waitKey(10);

        video_writer->write(result);
    }
#endif

    const auto section_4_frame = frame_pre_section * 2;
    const auto section_4_frame_1 = static_cast<int>(section_4_frame * 0.8);
#ifdef ENABLE_CHAPTER_1_SECTION_4
    std::cout << "Chapter 1 Section 4" << std::endl;
    auto center_x = static_cast<int>(
        CANVAS_WIDTH
        *
        (0.5)
    );
    auto center_y = static_cast<int>(
        CANVAS_HEIGHT
        *
        (0.5)
    );
    auto radius_max =
            std::sqrt(
                powf(static_cast<float>(CANVAS_WIDTH - CANVAS_CENTER_X), 2)
                +
                powf(static_cast<float>(CANVAS_HEIGHT - CANVAS_CENTER_Y), 2)
            );
    auto radius_start = 150 * RatioVideoScale;
    for (int i = 0; i < section_4_frame_1; ++i) {
        //        output_frame_log(chapter_index, 3, i, section_3_frame);

        auto radius = radius_start +
                      (
                          (radius_max - radius_start)
                          *
                          static_cast<float>(i) / static_cast<float>(section_4_frame_1)
                      );

        cl_kernel kernel_mask = program_mask.CreateKernel();

        auto light_source_x = (img_school_door_width / 2);
        auto light_source_y = static_cast<int>(-100 * RatioVideoScale);

        KernelSetArg_Image_Mask_Simple(
            kernel_mask,
            device_img_school_door_4channel, device_img_school_door_mask_output,
            img_school_door_width, img_school_door_height, 4,
            center_x, center_y, radius,
            1, 1,
            light_source_x, light_source_y,
            0, 0, 0, 150
        );
        CLKernelEnqueue(
            queue, kernel_mask,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_mask);

        // Channel Convert
        auto kernel_channel_1 = program_channel.CreateKernel();
        KernelSetArg_Image_Channel(
            kernel_channel_1,
            device_img_school_door_mask_output,
            device_img_school_door_frame_output,
            img_school_door_width, img_school_door_height,
            4, 3
        );
        CLKernelEnqueue(
            queue, kernel_channel_1,
            2, global_work_size
        );
        clFinish(queue);
        clReleaseKernel(kernel_channel_1);

        OpenCLMemcpyFromDevice(
            queue,
            result.data,
            device_img_school_door_frame_output,
            img_school_door_width * img_school_door_height * 3 * sizeof(uchar)
        );

        //        cv::imshow("result", result);
        //        cv::waitKey(10);

        video_writer->write(result);
    }

    for (int i = section_4_frame_1; i < section_4_frame; ++i) {
        video_writer->write(result);
    }
#endif

    clReleaseMemObject(device_img_school_door_4channel);
    clReleaseMemObject(device_img_school_door_mask_output);
    clReleaseMemObject(device_img_school_door_frame_output);

    clReleaseCommandQueue(queue);
    return result;
}
