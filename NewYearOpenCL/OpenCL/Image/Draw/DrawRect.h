// Draw Rect On Image
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_DRAW_RECT_H
#define NEW_YEAR_OPENCL_DRAW_RECT_H

#include "../../Include/OpenCLInclude.h"
#include "../../Include/OpenCLError.h"
#include "../../Include/OpenCLFlow.h"
#include "../../Include/OpenCLProgram.h"

#include "../../../OpenCV/Include/OpenCVInclude.h"

OpenCLProgram CLCreateProgram_Draw_Rect(cl_context context, cl_device_id device);

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
);

#endif //NEW_YEAR_OPENCL_DRAW_RECT_H
