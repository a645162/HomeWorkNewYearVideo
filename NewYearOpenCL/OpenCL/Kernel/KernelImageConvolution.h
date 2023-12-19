// OpenCL Kernel Function of Image Convolution
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_CONVOLUTION_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_CONVOLUTION_H

const char *cl_kernel_convolution = R"(
// Convolution.cl

// OpenCL Kernel Function of 2-Dim Convolution
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

// Define OpenCL kernel for 2-Dim convolution
__kernel void convolution2Dim(__global uchar *input, __global uchar *output,
                              int height, int width, int channels,
                              __global float *conv_kernel, int kernelSize,
                              int padSize) {

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int c = get_global_id(2);

    if (x < width && y < height && c < channels) {
		
		output[(y * width + x) * channels + c] = 0;

        if (c == 3) {
            output[(y * width + x) * channels + c] =
                input[(y * width + x) * channels + c];
            return;
        }

        float current_channel_result = 0.0f;

        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                int imageX = (int)x + i - padSize;
                int imageY = (int)y + j - padSize;

                if (imageX >= 0 && imageX < width && imageY >= 0 &&
                    imageY < height) {
                    current_channel_result +=
                        conv_kernel[j * kernelSize + i] *
                        (float)(input[(imageY * width + imageX) * channels +
                                      c]);
                }
            }
        }

        output[(y * width + x) * channels + c] =
            (uchar)(current_channel_result);
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_CONVOLUTION_H
