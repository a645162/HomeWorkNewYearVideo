// Image Crop
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageCrop.h"

#include "../Kernel/KernelImageCrop.h"


#include <iostream>

#include "../../Author/Author.h"

#include "../Devices/OpenCLDevices.h"

#include <opencv2/opencv.hpp>

OpenCLProgram CLCreateProgram_Image_Crop(cl_context context, cl_device_id device) {
    return {
        context,
        device,
        "cropImage",
        cl_kernel_crop
    };
}

void KernelSetArg_Image_Crop(
    cl_kernel kernel,
    cl_mem devSrc,
    cl_mem devDst,
    int input_width, int input_height,
    int output_width, int output_height,
    int x1, int y1,
    int x2, int y2,
    int channels
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devSrc);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &devDst);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &input_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &output_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &output_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &x2);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &y2);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);
}
