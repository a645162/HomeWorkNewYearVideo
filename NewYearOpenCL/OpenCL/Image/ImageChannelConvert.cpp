// Image Channel Convert
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageChannelConvert.h"

#include "../Kernel/KernelImageChannel.h"

OpenCLProgram CLCreateProgram_Image_Channel(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "ImageChannelConvert",
            cl_kernel_channel
    };
}

void KernelSetArg_Image_Channel(
        cl_kernel kernel,
        cl_mem device_image_input,
        cl_mem device_image_output,
        int image_width, int image_height,
        int src_channels, int dst_channels
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_input);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_image_output);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &src_channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &dst_channels);
}
