// Image Resize
// Created by Haomin Kong on 23-12-12.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageResize.h"

#include "../Kernel/KernelImageResize.h"

#include "../../OpenCV/Include/OpenCVInclude.h"

OpenCLProgram CLCreateProgram_Image_Resize(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "resizeImage",
            cl_kernel_resize_image
    };
}

void KernelSetArg_Image_Resize(
        cl_kernel kernel,
        cl_mem devSrc, cl_mem devDst,
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight,
        int channels
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &srcWidth);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &srcHeight);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &dstWidth);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &dstHeight);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
}

[[maybe_unused]] unsigned int calculateNewHeightByNewWidth(
        unsigned int width,
        unsigned int height,
        unsigned int newWidth
) {
    return static_cast<int>(
            roundf(
                    static_cast<float>(height) * static_cast<float>(newWidth)
                    /
                    static_cast<float>(width)
            )
    );
}

[[maybe_unused]] unsigned int calculateNewWidthByNewHeight(
        unsigned int width,
        unsigned int height,
        unsigned int newHeight
) {
    return static_cast<int>(
            roundf(
                    static_cast<float>(width) * static_cast<float>(newHeight)
                    /
                    static_cast<float>(height)
            )
    );
}
