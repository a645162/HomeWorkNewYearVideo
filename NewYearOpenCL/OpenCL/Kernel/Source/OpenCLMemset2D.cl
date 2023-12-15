__kernel void OpenCLMemset2D(__global uchar *target_device_memory, int width,
                             int height, const uchar value) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        target_device_memory[y * width + x] = value;
    }
}
