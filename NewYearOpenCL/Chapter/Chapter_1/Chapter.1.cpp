// Chapter.1
// Created by Haomin Kong on 23-12-17.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "Chapter.1.h"
#include "../../OpenCL/Image/ImageResize.h"
#include "../../OpenCL/Image/ImageChannelConvert.h"
#include "../../OpenCL/Image/ImageMerge.h"
#include "../../Utils/ProgramLog.h"

extern float RatioVideoScale;
extern float RatioVideoFrame;

extern int CANVAS_WIDTH, CANVAS_HEIGHT;
extern int CANVAS_CENTER_X, CANVAS_CENTER_Y;
extern int FRAME_RATE;

const int chapter_index = 1;

void chapter_1(
        cl_context context, cl_device_id device,
        int max_frame, cv::VideoWriter video_writer
) {

    std::cout << "Chapter 1" << std::endl;
    const int frame_pre_section = max_frame / 7;

    // Section 1

    auto logo_start_size = static_cast<int>(std::max(CANVAS_WIDTH, CANVAS_HEIGHT) * 1.2);
    auto logo_min_size = static_cast<int>(400 * RatioVideoScale);

    // Read input image
    cv::Mat shmtu_logo = cv::imread("../Resources/Image/shmtu_logo.png", cv::IMREAD_UNCHANGED);

    // Create White Canvas
    cv::Mat canvas = cv::Mat::zeros(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC4);
    canvas.setTo(cv::Scalar(255, 255, 255, 255));

    int logo_src_width = shmtu_logo.cols;
    int logo_src_height = shmtu_logo.rows;
    int logo_src_channels = shmtu_logo.channels();

    cl_command_queue queue = CLCreateCommandQueue(context, device);

    OpenCLProgram program_resize = CLCreateProgram_Image_Resize(context, device);
    OpenCLProgram program_merge = CLCreateProgram_Image_Merge(context, device);
    OpenCLProgram program_channel = CLCreateProgram_Image_Channel(context, device);

    cl_mem device_canvas_ori = OpenCLMalloc(
            context,
            CANVAS_WIDTH * CANVAS_HEIGHT * logo_src_channels * sizeof(uchar),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            canvas.data
    );

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

    cv::Mat result(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC(3));

    const auto section_1_frame = frame_pre_section * 2;
    for (int i = 0; i < static_cast<int>(section_1_frame * 0.8); ++i) {
        output_frame_log(chapter_index, 1, i, section_1_frame);
        auto current_size =
                logo_start_size -
                static_cast<int>(
                        (
                                static_cast<float>(i) / static_cast<float>(section_1_frame * 0.8 - 0)
                        )
                        *
                        (logo_start_size - logo_min_size)
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
                device_canvas_ori, device_resized_logo, device_merge_target,
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

        video_writer.write(result);
    }

    for (int i = static_cast<int>(section_1_frame * 0.8); i < section_1_frame; ++i) {
        output_frame_log(chapter_index, 1, i, section_1_frame);
        video_writer.write(result);
    }

    clReleaseMemObject(device_canvas_ori);
    clReleaseMemObject(device_logo_ori);
    clReleaseMemObject(device_merge_target);
    clReleaseMemObject(device_output_3channel);

    const auto section_2_frame = frame_pre_section * 2;
    cv::Mat img_school_door = cv::imread("../Resources/Image/IMG_3493.jpg", cv::IMREAD_UNCHANGED);

    const auto section_3_frame = frame_pre_section * 1;
    const auto section_4_frame = frame_pre_section * 2;


    clReleaseCommandQueue(queue);
}