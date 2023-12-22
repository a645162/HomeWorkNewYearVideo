// ImageReverseColor.cl

// OpenCL Kernel Function of Image Reverse Color
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void resizeImage(__global const uchar *src, __global uchar *dst,
                          int width, int height, int channels) {
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    if (x < width && y < height) {
        int index = (y * width + x) * channels;

        dst[index + 0] = 255 - src[index + 0];
        dst[index + 1] = 255 - src[index + 1];
        dst[index + 2] = 255 - src[index + 2];

        if (channels == 4) {
            dst[index + 3] = src[index + 3];
        }
    }
}
