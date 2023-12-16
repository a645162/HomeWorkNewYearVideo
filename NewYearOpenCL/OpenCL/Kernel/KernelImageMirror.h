// OpenCL Kernel Function of Image Mirror
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_MIRROR_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_MIRROR_H

const char *cl_kernel_mirror = R"(
// OpenCL Kernel Function of Image Mirror
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void ImageMirror(__global const uchar *inputImage,
                          __global uchar *outputImage, int width, int height,
                          int channels, int type) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        // Mirror horizontally
        int from_x = x, from_y = y;
        if (x > width / 2) {
            if (type != 0) {
                from_x = width - 1 - x;
            }
        } else {
            if (type == 0) {
                from_x = width - 1 - x;
            }
        }
        for (int c = 0; c < channels; ++c) {
            outputImage[(y * width + x) * channels + c] =
                inputImage[(from_y * width + from_x) * channels + c];
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_MIRROR_H
