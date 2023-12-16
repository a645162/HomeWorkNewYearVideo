// OpenCL Kernel Function of Image Resize
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_RESIZE_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_RESIZE_H

const char *cl_kernel_resize_image = R"(
// ImageResize.cl

// OpenCL Kernel Function of Image Resize
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void resizeImage(__global const uchar *src, __global uchar *dst,
                          int srcWidth, int srcHeight, int dstWidth,
                          int dstHeight, int channels) {
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    if (x < dstWidth && y < dstHeight) {
        float scaleX = (float)srcWidth / dstWidth;
        float scaleY = (float)srcHeight / dstHeight;
        float srcX = x * scaleX;
        float srcY = y * scaleY;
        int x1 = (int)srcX;
        int y1 = (int)srcY;
        int x2 = x1 + 1;
        int y2 = y1 + 1;
        float xWeight = srcX - x1;
        float yWeight = srcY - y1;
        for (int c = 0; c < channels; ++c) {
            float topLeft = src[(y1 * srcWidth + x1) * channels + c];
            float topRight = src[(y1 * srcWidth + x2) * channels + c];
            float bottomLeft = src[(y2 * srcWidth + x1) * channels + c];
            float bottomRight = src[(y2 * srcWidth + x2) * channels + c];
            float topInterpolation =
                topLeft * (1 - xWeight) + topRight * xWeight;
            float bottomInterpolation =
                bottomLeft * (1 - xWeight) + bottomRight * xWeight;
            dst[(y * dstWidth + x) * channels + c] =
                topInterpolation * (1 - yWeight) +
                bottomInterpolation * yWeight;
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_RESIZE_H
