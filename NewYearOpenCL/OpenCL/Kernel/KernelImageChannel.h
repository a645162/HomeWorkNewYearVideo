// OpenCL Kernel Function of Image Channel
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_CHANNEL_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_CHANNEL_H

const char *cl_kernel_channel = R"(
// OpenCL Kernel Function of Image Channel Convert
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void ImageChannelConvert(__global const uchar *inputImage,
                                  __global uchar *outputImage, int width,
                                  int height, int src_channels,
                                  int dst_channels) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {

        for (int c = 0; c < dst_channels; c++) {
            int dst_index = (y * width + x) * dst_channels + c;
            int src_index = (y * width + x) * src_channels + c;
            outputImage[dst_index] = inputImage[src_index];
        }

        // big to small
        // 3,4->1 [index_0]
        // 4->3 [index_0, index_1, index_2]
        // small to big
        // 1->3,4 [index_0, index_0, index_0, 255]
        // 3->4 [index_0, index_1, index_2, 255]
        if (dst_channels > src_channels) {
            int src_index = (y * width + x) * src_channels;
            for (int c = src_channels; c < dst_channels; c++) {
                int dst_index = (y * width + x) * dst_channels + c;
                if (c == 3) {
                    outputImage[dst_index] = 255;
                } else {
                    outputImage[dst_index] = inputImage[src_index];
                }
            }
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_CHANNEL_H
