__kernel void generateGradientColor(__global uchar *gradientColor,
                                    const int length, const int startR,
                                    const int startG, const int startB,
                                    const int endR, const int endG,
                                    const int endB, const unsigned int channels,
                                    const uchar alpha) {
    int idx = get_global_id(0);

    if (idx < length) {
        float t = (float)idx / (float)(length - 1);

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
