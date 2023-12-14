//
// Created by konghaomin on 23-12-13.
//

#ifndef NEW_YEAR_OPENCL_KERNEL_CONVOLUTION_H
#define NEW_YEAR_OPENCL_KERNEL_CONVOLUTION_H

const char *cl_kernel_convolution = R"(
// Convolution.cl
// Define OpenCL kernel for 2D convolution
__kernel void convolution2D(__global uchar *input, __global uchar *output,
                            int height, int width, int channels,
                            __global float *conv_kernel, int kernelSize,
                            int padSize) {
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int c = get_global_id(2);

    if (x < width && y < height) {

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
                        conv_kernel[i * kernelSize + j] *
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

#endif //NEW_YEAR_OPENCL_KERNEL_CONVOLUTION_H
