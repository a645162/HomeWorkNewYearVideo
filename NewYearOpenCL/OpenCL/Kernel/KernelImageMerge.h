// OpenCL Kernel Function of Image Merge
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_KERNEL_IMAGE_MERGE_H
#define NEW_YEAR_OPENCL_KERNEL_IMAGE_MERGE_H

const char *cl_kernel_merge = R"(
// OpenCL Kernel Function of Image Merge
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void mergeImages(__global const uchar *image1,
                          __global const uchar *image2, __global uchar *output,
                          const int image1_width, const int image1_height,
                          const int image1_channels, const int image2_target_x,
                          const int image2_target_y, const int image2_width,
                          const int image2_height, const int image2_channels,
                          const int image2_alpha) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < image1_width && y < image1_height) {
        const int index = (y * image1_width + x) * image1_channels;

        if ((x >= image2_target_x && x < (image2_target_x + image2_width)) &&
            (y >= image2_target_y && y < (image2_target_y + image2_height))) {
            int image2_X = x - image2_target_x;
            int image2_Y = y - image2_target_y;

            // Using 4 channels for BGRA
            int image2_index =
                (image2_Y * image2_width + image2_X) * image2_channels;

            const float user_set_alpha = image2_alpha / 255.0f;
            if (image2_channels == 4) {
                // Normalize alpha to range [0, 1]
                const float alpha = image2[image2_index + 3] / 255.0f;
                const float alpha_final = alpha * user_set_alpha;

                output[index] =
                    convert_uchar_rte(image1[index] * (1.0f - alpha_final) +
                                      image2[image2_index] * alpha_final);
                output[index + 1] =
                    convert_uchar_rte(image1[index + 1] * (1.0f - alpha_final) +
                                      image2[image2_index + 1] * alpha_final);
                output[index + 2] =
                    convert_uchar_rte(image1[index + 2] * (1.0f - alpha_final) +
                                      image2[image2_index + 2] * alpha_final);
            } else {
                for (int c = 0; c < min(image1_channels, image2_channels);
                     ++c) {
                    output[index + c] = convert_uchar_rte(
                        image1[index + c] * (1.0f - user_set_alpha) +
                        image2[image2_index + c] * user_set_alpha);
                }
            }
        } else {
            for (int c = 0; c < image1_channels; ++c) {
                output[index + c] = image1[index + c];
            }
        }

        // Copy image1 alpha channel to output if it exists
        if (image1_channels == 4) {
            output[index + 3] = image1[index + 3];
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_IMAGE_MERGE_H
