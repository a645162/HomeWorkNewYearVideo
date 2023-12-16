// OpenCL Kernel Function of Generate Gradient Image
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_GENERATE_GRADIENT_IMAGE_H
#define NEW_YEAR_OPENCL_KERNEL_GENERATE_GRADIENT_IMAGE_H

const char *cl_kernel_generate_gradient_image = R"(
// OpenCL Kernel Function of Generate Gradient Image
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void generateGradientImage(__global uchar *gradientImage,
                                    __global const uchar *gradientColor,
                                    const int colorCount, const int width,
                                    const int height, const int center_x,
                                    const int center_y, const float maxR,
                                    const uchar channels, const uchar alpha) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        float r =
            sqrt(pow((float)(x - center_x), 2) + pow((float)(y - center_y), 2));
        int idx = convert_int_rte(r / maxR * (float)(colorCount - 1));

        int image_index = channels * (y * width + x);
        int color_index = channels * idx;

        gradientImage[image_index + 0] = gradientColor[color_index + 0];
        gradientImage[image_index + 1] = gradientColor[color_index + 1];
        gradientImage[image_index + 2] = gradientColor[color_index + 2];

        if (channels == 4) {
            gradientImage[image_index + 3] = alpha;
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_GENERATE_GRADIENT_IMAGE_H
