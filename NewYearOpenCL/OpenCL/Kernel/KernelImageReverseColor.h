// OpenCL Kernel Function of Image Reverse Color
// Created by Haomin Kong on 23-12-25.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_REVERSE_COLOR_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_REVERSE_COLOR_H

const char* cl_kernel_reverse_color = R"(
// ImageReverseColor.cl

// OpenCL Kernel Function of Image Reverse Color
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void ImageReverseColor(__global const uchar *src, __global uchar *dst,
                                int width, int height, int channels) {
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    if (x < width && y < height) {
        int index = (y * width + x) * channels;

        for (int c = 0; c < max(channels, 3); ++c) {
            dst[index + c] = 255 - src[index + c];
        }

        if (channels == 4) {
            dst[index + 3] = src[index + 3];
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_REVERSE_COLOR_H
