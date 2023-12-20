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
        int index = (y * width + x) * channels;
        uchar value = 0;
        uchar avg_value = (inputImage[index + 0] + inputImage[index + 1] +
                           inputImage[index + 2]) /
                          3;

        if (avg_value > threshold) {
            value = 255;
        }

        if (reverse_color == 1) {
            value = 255 - value;
        }

        for (int i = 0; i < channels; i++) {
            outputImage[index + i] = value;
        }
        // if (channels == 4) {
        //     outputImage[index + 3] = inputImage[index + 3];
        // }
    }
}
