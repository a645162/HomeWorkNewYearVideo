// OpenCL Kernel Function of Image Mask
// Created by Haomin Kong on 23-12-12.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_MASK_IMAGE_KERNEL_H
#define NEW_YEAR_OPENCL_MASK_IMAGE_KERNEL_H

const char* cl_kernel_mask_image_circle = R"(
// MaskImageCircle.cl

// OpenCL Kernel Function of Mask Image with Circle and Focus Light Effect
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void
MaskImageCircle(const __global uchar *input, __global uchar *output,
                const int width, const int height, const int channels,
                const int centerX, const int centerY, const float radius,
                int clean_up_alpha, int focus_lamp, const int light_source_x,
                const int light_source_y, const float m_1, const float m_2,
                const float max_distance, uchar focus_color_b,
                uchar focus_color_g, uchar focus_color_r, uchar color_alpha) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // if(x==0&&y==0){
    //     printf("clean_up_alpha:%d\n",clean_up_alpha);

    //     printf("focus_lamp:%d\n",focus_lamp);

    //     printf("light_source_x:%d\n",light_source_x);
    //     printf("light_source_y:%d\n",light_source_y);

    //     printf("m_1:%f\n",m_1);
    //     printf("m_2:%f\n",m_2);
    //     printf("max_distance:%f\n",max_distance);

    //     printf("focus_color_x:%d\n",focus_color_x);
    //     printf("focus_color_y:%d\n",focus_color_y);
    //     printf("focus_color_z:%d\n",focus_color_z);
    //     printf("color_alpha:%d\n",color_alpha);
    // }

    if (x < width && y < height) {
        const int index = (y * width + x) * channels;

        const float distance =
            sqrt(pow((float)(x - centerX), 2) + pow((float)(y - centerY), 2));

        if (distance > radius) {
            output[index + 3] = 0;

            if (focus_lamp != 0) {
                const float x_1 =
                    m_1 * (float)(y - light_source_y) + (float)light_source_x;
                const float x_2 =
                    m_2 * (float)(y - light_source_y) + (float)light_source_x;

                const float x_left = fmin(x_1, x_2);
                const float x_right = fmax(x_1, x_2);

                const float distance_current =
                    sqrt(pow((float)(x - light_source_x), 2) +
                         pow((float)(y - light_source_y), 2));

                if (x_left <= (float)x && (float)x <= x_right &&
                    distance_current <= max_distance) {
                    const float color_alpha_rate = (float)color_alpha / 255.0f;

                    output[index + 0] = convert_uchar_sat_rte(
                        convert_float(input[index + 0]) *
                            (1 - color_alpha_rate) +
                        convert_float(focus_color_b) * color_alpha_rate);
                    output[index + 1] = convert_uchar_sat_rte(
                        convert_float(input[index + 1]) *
                            (1 - color_alpha_rate) +
                        convert_float(focus_color_g) * color_alpha_rate);
                    output[index + 2] = convert_uchar_sat_rte(
                        convert_float(input[index + 2]) *
                            (1 - color_alpha_rate) +
                        convert_float(focus_color_r) * color_alpha_rate);
                    if (channels == 4) {
                        output[index + 3] = color_alpha;
                    }
                    return;
                }
            }

            if (clean_up_alpha) {
                output[index + 0] = 0;
                output[index + 1] = 0;
                output[index + 2] = 0;
            }
        } else {
            output[index] = input[index];
            output[index + 1] = input[index + 1];
            output[index + 2] = input[index + 2];
            output[index + 3] = input[index + 3];
        }
    }
}
)";

#endif //NEW_YEAR_OPENCL_MASK_IMAGE_KERNEL_H
