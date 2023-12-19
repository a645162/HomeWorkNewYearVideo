// OpenCL Kernel Function of Memset 2D Image Each Channel to a Value.
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void OpenCLMemset2D(__global uchar *target_device_memory, int width,
                             int height, int channel, const uchar value) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int c = get_global_id(2);

    if (x < width && y < height && c < channel) {
        const int index_start = (y * width + x) * channel;
        target_device_memory[index_start + c] = value;
    }
}
