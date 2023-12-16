// OpenCL Kernel Function of Image Rotate
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_ROTATE_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_ROTATE_H

const char *cl_kernel_rotate = R"(
// OpenCL Kernel Function of Image Rotate
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

#define PI 3.14159265

__kernel void rotateImage(__global const uchar *input, __global uchar *output,
                          int width, int height, int channels, float angle) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float radians = angle * PI / 180.0;
    float cosVal = cos(radians);
    float sinVal = sin(radians);

    int centerX = width / 2;
    int centerY = height / 2;

    int rotatedX = convert_int_rte(cosVal * (x - centerX) -
                                   sinVal * (y - centerY) + centerX);
    int rotatedY = convert_int_rte(sinVal * (x - centerX) +
                                   cosVal * (y - centerY) + centerY);

    if (rotatedX >= 0 && rotatedX < width && rotatedY >= 0 &&
        rotatedY < height) {
        int inputIndex = (rotatedY * width + rotatedX) * channels;
        int outputIndex = (y * width + x) * channels;

        for (int c = 0; c < channels; ++c) {
            output[outputIndex + c] = input[inputIndex + c];
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_ROTATE_H
