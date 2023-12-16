// Draw Rect On Image
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "DrawRect.h"

#include "../../Kernel/KernelDrawRect.h"

OpenCLProgram CLCreateProgram_Draw_Rect(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "drawRectangle",
            cl_kernel_draw_rect
    };
}

void KernelSetArg_Draw_Rect(
        cl_kernel kernel,
        cl_mem device_image,
        int width, int height,
        int x1, int y1,
        int x2, int y2,
        int thickness,
        uchar board_color_r, uchar board_color_g, uchar board_color_b,
        uchar fill_color_r, uchar fill_color_g, uchar fill_color_b,
        int channels, int fill,
        int sine_waves_board, float frequency
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y1);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x2);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y2);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &thickness);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &board_color_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &board_color_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &board_color_b);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &fill_color_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &fill_color_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &fill_color_b);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &fill);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &sine_waves_board);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &frequency);
}
