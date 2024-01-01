// OpenCL Kernel Function of Image Channel Convert
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void ImageChannelConvert(__global const uchar *inputImage,
                                  __global uchar *outputImage, int width,
                                  int height, int src_channels,
                                  int dst_channels) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height) {

        float alpha_rate = 1.0f;

        if (src_channels == 4 && dst_channels < 4) {
            alpha_rate =
                ((float)inputImage[(y * width + x) * src_channels + 3]) /
                255.0f;
        }

		int minChannel = min(src_channels, dst_channels);
        for (int c = 0; c < minChannel; c++) {
            const int dst_index = (y * width + x) * dst_channels + c;
            const int src_index = (y * width + x) * src_channels + c;
            outputImage[dst_index] =
                (uchar)((float)(inputImage[src_index]) * alpha_rate);
        }

        // big to small
        // 3,4->1 [index_0]
        // 4->3 [index_0, index_1, index_2]
        // small to big
        // 1->3,4 [index_0, index_0, index_0, 255]
        // 3->4 [index_0, index_1, index_2, 255]
        if (dst_channels > src_channels) {
            // int src_index = (y * width + x) * src_channels;
            const int dst_index_start = (y * width + x) * dst_channels;
            for (int c = src_channels; c < dst_channels; c++) {
                const int dst_index = dst_index_start + c;
                if (c == 3) {
                    outputImage[dst_index] = 255;
                } else {
                    outputImage[dst_index] = outputImage[dst_index_start];
                }
            }
        }
    }
}
