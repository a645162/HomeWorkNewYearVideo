//
// Created by 孔昊旻 on 2023/12/15.
//

#ifndef NEW_YEAR_OPENCL_KERNEL_MIRROR_H
#define NEW_YEAR_OPENCL_KERNEL_MIRROR_H

const char *cl_kernel_mirror = R"(
__kernel void ImageMirror(__global const uchar3 *inputImage,
                          __global uchar3 *outputImage, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width / 2 && y < height) {
        // Mirror horizontally
        outputImage[y * width + x] = inputImage[y * width + (width - 1 - x)];
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_MIRROR_H
