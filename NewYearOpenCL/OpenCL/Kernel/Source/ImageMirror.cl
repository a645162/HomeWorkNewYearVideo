__kernel void ImageMirror(__global const uchar3 *inputImage,
                          __global uchar3 *outputImage, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        // Mirror horizontally
        outputImage[y * width + x] = inputImage[y * width + (width - 1 - x)];
    }
}
