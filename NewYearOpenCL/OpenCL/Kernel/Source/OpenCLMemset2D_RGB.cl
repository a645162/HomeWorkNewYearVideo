// OpenCL Kernel Function of Memset 2D Image RGB(A)
// Author: Haomin Kong.
// https://github.com/a645162/HomeWorkNewYearVideo

__kernel void OpenCLMemset2D(__global uchar *target_device_memory, int width,
                             int height, int channel, const uchar r,
                             const uchar g, const uchar b, const uchar a) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height) {
        const int index_start = (y * width + x) * channel;
        target_device_memory[index_start + 0] = r;
        target_device_memory[index_start + 1] = g;
        target_device_memory[index_start + 2] = b;
        if (channel == 4) {
            target_device_memory[index_start + 3] = a;
        }
    }
}
