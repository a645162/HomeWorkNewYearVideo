// Convolution.cl
// Define OpenCL kernel for 2D convolution
__kernel void convolution2D(__global uchar *input, __global uchar *output,
                            int height, int width, int channels,
                            __global float *conv_kernel, int kernelSize,
                            int padSize) {
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t c = get_global_id(2);

  if (x < width - kernelSize + 1 && y < height - kernelSize + 1 &&
      c < channels) {

    if (c == 3) {
      output[(y * width + x) * channels + c] =
          input[(y * width + x) * channels + c];
      return;
    }

    float current_channel_result = 0.0f;

    for (int i = 0; i < kernelSize; ++i) {
      for (int j = 0; j < kernelSize; ++j) {
        int imageX = (int)x + i - padSize;
        int imageY = (int)y + j - padSize;

        if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
          current_channel_result +=
              conv_kernel[i * kernelSize + j] *
              (float)(input[(imageY * width + imageX) * channels + c]);
        }
      }
    }

    output[(y * width + x) * channels + c] = (uchar)(current_channel_result);
  }
}
