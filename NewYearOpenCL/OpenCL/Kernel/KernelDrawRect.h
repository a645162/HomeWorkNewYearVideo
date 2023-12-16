// OpenCL Kernel Function of Draw Rect
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_DRAW_RECT_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_DRAW_RECT_H

const char *cl_kernel_draw_rect = R"(
// OpenCL Kernel Function of Draw Rectangle
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void drawRectangle(__global uchar *d_image, const int width,
                            const int height, const int x_1, const int y_1,
                            const int x_2, const int y_2, const int thickness,
                            uchar board_color_r, uchar board_color_g,
                            uchar board_color_b, uchar fill_color_r,
                            uchar fill_color_g, uchar fill_color_b,
                            const int channels, const int fill,
                            const int sine_waves_board, const float frequency) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int x1 = min(x_1, x_2);
    const int y1 = min(y_1, y_2);

    const int x2 = max(x_1, x_2);
    const int y2 = max(y_1, y_2);

    const int index = (y * width + x) * channels;

    if ((x1 <= x && x <= x2) && (y1 <= y && y <= y2)) {
        // Draw Area

        // Fill the rectangle
        if (thickness <= 0) {
            if (fill != 0) {
                d_image[index + 0] = fill_color_b;
                d_image[index + 1] = fill_color_g;
                d_image[index + 2] = fill_color_r;
            }
            return;
        }

        const int fill_x1 = x1 + thickness;
        const int fill_y1 = y1 + thickness;

        const int fill_x2 = x2 - thickness;
        const int fill_y2 = y2 - thickness;

        if ((fill_x1 <= x && x <= fill_x2) && (fill_y1 <= y && y <= fill_y2)) {
            // Fill Area
            if (fill != 0) {
                d_image[index + 0] = fill_color_b;
                d_image[index + 1] = fill_color_g;
                d_image[index + 2] = fill_color_r;
            }
        } else {
            // Bord Area
            uchar current_board_color_r = board_color_r;
            uchar current_board_color_g = board_color_g;
            uchar current_board_color_b = board_color_b;

            if (sine_waves_board != 0) {
                // Draw sine waves along the four edges of the rectangle
                float sineValue = 0.0;

                // X Fixed Vertical Line
                if ((x1 <= x && x <= x1 + thickness) ||
                    (x2 - thickness <= x && x <= x2)) {
                    sineValue = sin((float)y * frequency);
                }

                // Y Fixed Horizontal Line
                if ((y1 <= y && y <= y1 + thickness) ||
                    (y2 - thickness <= y && y <= y2)) {
                    sineValue = sin((float)x * frequency);
                }

                // Normalize the sine value to the range [0, 1]
                sineValue += 1.0;
                sineValue /= 2.0;

                const int color_x =
                    (int)(sineValue * (float)current_board_color_b);
                const int color_y =
                    (int)(sineValue * (float)current_board_color_g);
                const int color_z =
                    (int)(sineValue * (float)current_board_color_r);

                current_board_color_b = (uchar)(color_x);
                current_board_color_g = (uchar)(color_y);
                current_board_color_r = (uchar)(color_z);
            }

            d_image[index + 0] = current_board_color_b;
            d_image[index + 1] = current_board_color_g;
            d_image[index + 2] = current_board_color_r;
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_DRAW_RECT_H
