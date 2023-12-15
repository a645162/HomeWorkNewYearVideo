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
#include "OpenCL/Image/ImageMirror.h"
#include "OpenCL/Image/ImageResize.h"
#include "OpenCL/Image/ImageConvolution.h"
#include "OpenCL/Image/ImageCrop.h"
#include "OpenCL/Image/ImageRotate.h"
#include "OpenCL/Image/ImageMask.h"

int main() {

#ifndef DEBUG_MODE
    KHM::sayHello();
#endif

    cl_device_id device = UserSelectDevice();

    cl_context context =
            CLCreateContext(device);

//    mask_video_demo(context, device);

//    rotate_demo(context, device);

//    crop_demo(context, device);

//    conv_demo(context, device);

    // resize demo
//    resize_demo(context, device);

    // convert channel demo
//    convert_channel_demo(context, device);

    // merge demo
//    merge_demo(context, device);

//    convert_gray_demo(context, device);

    // Image Mirror Horizontal
//    mirror_demo(context, device);

    clReleaseContext(context);

    return 0;
}