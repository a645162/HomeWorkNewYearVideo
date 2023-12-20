// OpenCL Kernel Function of Image Binaryzation
// Author: Haomin Kong.
// 2023/12/20
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void ImageBinaryzation(__global const uchar *inputImage,
                                  __global uchar *outputImage, int width,
                                  int height, int channels, uchar threshold,
                                  int reverse_color) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height) {
        // Only handle 3 channels
        int max_channel = max(channels, 3);
        int index = (y * width + x) * channels;
        uchar value = 0;

        for (int i = 0; i < max_channel; i++) {
            if (inputImage[index + i] > threshold) {
                value = 255;
                break;
            }
        }

        if (reverse_color == 1) {
            value = 255 - value;
        }

        for (int i = 0; i < max_channel; i++) {
            outputImage[index + i] = value;
        }
    }
}
