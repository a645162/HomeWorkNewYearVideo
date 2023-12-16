// Generate Gradient Image
// Created by Haomin Kong on 23-12-15.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "GenerateGradientImage.h"

#include "../../Kernel/KernelGenerateGradientColor.h"
#include "../../Kernel/KernelGenerateGradientImage.h"

OpenCLProgram CLCreateProgram_Generate_GradientColor(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "generateGradientColor",
            cl_kernel_generate_gradient_color
    };
}

OpenCLProgram CLCreateProgram_Generate_GradientImage(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "generateGradientImage",
            cl_kernel_generate_gradient_image
    };
}


void KernelSetArg_Generate_GradientColor(
        cl_kernel kernel,
        cl_mem device_gradient_color,
        int color_count,
        uchar start_r, uchar start_g, uchar start_b,
        uchar end_r, uchar end_g, uchar end_b,
        uchar channels, uchar alpha
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_gradient_color);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &color_count);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &start_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &start_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &start_b);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &end_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &end_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &end_b);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &alpha);
}

void KernelSetArg_Generate_GradientImage(
        cl_kernel kernel,
        cl_mem device_gradient_image,
        cl_mem device_gradient_color,
        int color_count,
        int image_width, int image_height,
        int center_x, int center_y, float max_r,
        uchar channels, uchar alpha
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_gradient_image);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_gradient_color);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &color_count);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &image_height);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &center_x);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &center_y);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &max_r);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &channels);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &alpha);
}
