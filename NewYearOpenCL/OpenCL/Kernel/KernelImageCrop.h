// OpenCL Kernel Function of Image Crop
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_CROP_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_CROP_H

const char *cl_kernel_crop = R"(
// OpenCL Kernel Function of Image Crop
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void cropImage(__global const uchar *input, __global uchar *output,
                        int inputWidth, int inputHeight, int outputWidth,
                        int outputHeight, int x1, int y1, int x2, int y2,
                        const int channels) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < outputWidth && y < outputHeight) {
        int inputX = x + x1;
        int inputY = y + y1;

        int inputIndex = (inputY * inputWidth + inputX) * channels;
        int outputIndex = (y * outputWidth + x) * channels;

        output[outputIndex + 0] = input[inputIndex + 0];
        output[outputIndex + 1] = input[inputIndex + 1];
        output[outputIndex + 2] = input[inputIndex + 2];

        if (channels == 4) {
            output[outputIndex + 3] = input[inputIndex + 3];
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_CROP_H
