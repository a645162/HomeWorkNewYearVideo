#define PI 3.14159265

__kernel void rotateImage(__global const uchar *input, __global uchar *output,
                          int width, int height, float angle) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float radians = angle * PI / 180.0;
    float cosVal = cos(radians);
    float sinVal = sin(radians);

    int centerX = width / 2;
    int centerY = height / 2;

    int rotatedX = convert_int_rte(cosVal * (x - centerX) -
                                   sinVal * (y - centerY) + centerX);
    int rotatedY = convert_int_rte(sinVal * (x - centerX) +
                                   cosVal * (y - centerY) + centerY);

    if (rotatedX >= 0 && rotatedX < width && rotatedY >= 0 &&
        rotatedY < height) {
        int inputIndex = (rotatedY * width + rotatedX) * 3;
        int outputIndex = (y * width + x) * 3;

        output[outputIndex] = input[inputIndex];
        output[outputIndex + 1] = input[inputIndex + 1];
        output[outputIndex + 2] = input[inputIndex + 2];
    }
}
