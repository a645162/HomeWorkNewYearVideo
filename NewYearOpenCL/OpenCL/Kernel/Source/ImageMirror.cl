__kernel void ImageMirror(__global const uchar *inputImage,
                          __global uchar *outputImage, int width, int height,
                          int channels, int type) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        // Mirror horizontally
        int from_x = x, from_y = y;
        if (x > width / 2) {
            if (type != 0) {
                from_x = width - 1 - x;
            }
        } else {
            if (type == 0) {
                from_x = width - 1 - x;
            }
        }
        for (int c = 0; c < channels; ++c) {
            outputImage[(y * width + x) * channels + c] =
                inputImage[(from_y * width + from_x) * channels + c];
        }
    }
}
