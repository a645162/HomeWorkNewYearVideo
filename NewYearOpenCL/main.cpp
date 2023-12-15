//
// Created by konghaomin on 23-12-12.
//

#ifndef DEBUG_MODE
#include "Author/Author.h"
#endif

#include <iostream>

#include "OpenCL/Devices/OpenCLDevices.h"

// Image Processing
#include "OpenCL/Image/ImageChannelConvert.h"
#include "OpenCL/Image/ImageMerge.h"
#include "OpenCL/Image/ImageGrayRGB.h"

int main() {

#ifndef DEBUG_MODE
    KHM::sayHello();
#endif

    cl_device_id device = UserSelectDevice();

    cl_context context =
            CLCreateContext(device);

    // convert channel demo
//    convert_channel_demo(context, device);

    // merge demo
//    merge_demo(context, device);

    convert_gray_demo(context, device);

    clReleaseContext(context);

    return 0;
}