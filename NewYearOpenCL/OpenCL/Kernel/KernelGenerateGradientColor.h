//
// Created by konghaomin on 23-12-13.
//

#ifndef NEW_YEAR_OPENCL_KERNEL_GENERATE_GRADIENT_COLOR_H
#define NEW_YEAR_OPENCL_KERNEL_GENERATE_GRADIENT_COLOR_H

const char *cl_kernel_generate_gradient_color = R"(
// GradientColorGenerate.cl
__kernel void generateGradientColor(__global uchar *gradientColor,
                                    const int colorCount, const uchar startR,
                                    const uchar startG, const uchar startB,
                                    const uchar endR, const uchar endG,
                                    const uchar endB, const uchar channels,
                                    const uchar alpha) {
    int idx = get_global_id(0);

    if (idx < colorCount) {
        float t = (float)idx / (float)(colorCount - 1);

        int color_index = channels * idx;

        uchar colorR =
            convert_uchar_rte((1 - t) * (float)startR + t * (float)endR);
        uchar colorG =
            convert_uchar_rte((1 - t) * (float)startG + t * (float)endG);
        uchar colorB =
            convert_uchar_rte((1 - t) * (float)startB + t * (float)endB);

        gradientColor[color_index + 0] = colorB;
        gradientColor[color_index + 1] = colorG;
        gradientColor[color_index + 2] = colorR;

        if (channels == 4) {
            gradientColor[color_index + 3] = alpha;
        }
    }
}

)";

#endif //NEW_YEAR_OPENCL_KERNEL_GENERATE_GRADIENT_COLOR_H
