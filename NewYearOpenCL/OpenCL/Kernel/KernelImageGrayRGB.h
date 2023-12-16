// OpenCL Kernel Function of Image Gray
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_GRAY_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_GRAY_H

const char *cl_kernel_gray = R"(
// OpenCL Kernel Function of Image Gray from RGB
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

#define CONVERT_TO_GRAY_AVG(r, g, b) (uchar)((r + g + b) / 3)

#define WEIGHTED_GRAY_WEIGHT_R 0.299
#define WEIGHTED_GRAY_WEIGHT_G 0.587
#define WEIGHTED_GRAY_WEIGHT_B 0.114

#define CONVERT_TO_WEIGHTED_GRAY(r, g, b)                                      \
    (uchar)(WEIGHTED_GRAY_WEIGHT_R * (r) + WEIGHTED_GRAY_WEIGHT_G * (g) +      \
            WEIGHTED_GRAY_WEIGHT_B * (b))

__kernel void convertToGrayRGB(__global const uchar *inputImage,
                               __global uchar *outputImage, int width,
                               int height, int channels, int type) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;

        // Calculate pixel index for each channel
        int blueIndex = index * channels;
        int greenIndex = blueIndex + 1;
        int redIndex = blueIndex + 2;
        int alphaIndex = blueIndex + 3;

        uchar r = inputImage[redIndex];
        uchar g = inputImage[greenIndex];
        uchar b = inputImage[blueIndex];

        uchar grayValue;

        if (type == 0) {
            grayValue = CONVERT_TO_GRAY_AVG(r, g, b);
        } else {
            grayValue = CONVERT_TO_WEIGHTED_GRAY(r, g, b);
        }

        if (channels == 4) {
            outputImage[alphaIndex] = inputImage[alphaIndex];
        } else if (channels != 3) {
            grayValue = 0;
        }

        // Set the same grayscale value for all channels
        outputImage[redIndex] = grayValue;
        outputImage[greenIndex] = grayValue;
        outputImage[blueIndex] = grayValue;
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_GRAY_H
