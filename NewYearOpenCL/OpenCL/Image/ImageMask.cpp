// Image Mask
// Created by Haomin Kong on 23-12-12.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "ImageMask.h"

#include "../Kernel/KernelMaskImage.h"

OpenCLProgram CLCreateProgram_Image_Mask(cl_context context, cl_device_id device) {
    return {
            context,
            device,
            "MaskImageCircle",
            cl_kernel_mask_image_circle
    };
}

void KernelSetArg_Image_Mask(
        cl_kernel kernel,
        cl_mem device_input, cl_mem device_output,
        int width, int height, int channels,
        int centerX, int centerY, float radius,
        int clean_up_alpha, int focus_lamp,
        int light_source_x, int light_source_y,
        float m_1, float m_2, float max_distance,
        uchar focus_color_b, uchar focus_color_g, uchar focus_color_r, uchar color_alpha
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_input);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_output);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channels);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &centerX);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &centerY);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &radius);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &clean_up_alpha);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &focus_lamp);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &light_source_x);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &light_source_y);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &m_1);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &m_2);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(float), &max_distance);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &focus_color_b);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &focus_color_g);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &focus_color_r);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(uchar), &color_alpha);
}

void KernelSetArg_Image_Mask_Simple(
        cl_kernel kernel,
        cl_mem device_input, cl_mem device_output,
        int width, int height, int channels,
        int centerX, int centerY, float radius,
        int clean_up_alpha, int focus_lamp,
        int light_source_x, int light_source_y,
        uchar focus_color_b, uchar focus_color_g, uchar focus_color_r, uchar color_alpha
) {
    const auto k_center =
            (static_cast<float>(centerY - light_source_y)) / (static_cast<float>(centerX - light_source_x));
    const auto angle_center = atanf(k_center);
    const auto distance_center = sqrtf(
            powf(static_cast<float>(light_source_x - centerX), 2) +
            powf(static_cast<float>(light_source_y - centerY), 2)
    );
    const auto max_distance = sqrtf(
            powf(distance_center, 2) - powf(radius, 2)
    );

    const auto angle_between_center = asinf(radius / distance_center);
    const auto angle_1 = angle_center - angle_between_center;
    const auto angle_2 = angle_center + angle_between_center;

    const auto m_1 = 1.0f / tanf(angle_1);
    const auto m_2 = 1.0f / tanf(angle_2);

    KernelSetArg_Image_Mask(
            kernel,
            device_input, device_output,
            width, height, channels,
            centerX, centerY, radius,
            clean_up_alpha, focus_lamp,
            light_source_x, light_source_y,
            m_1, m_2, max_distance,
            focus_color_b, focus_color_g, focus_color_r, color_alpha
    );
}
