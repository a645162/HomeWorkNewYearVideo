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
#include "OpenCL/Image/Generate/GenerateGradientImage.h"
#include "OpenCL/Image/Draw/DrawRect.h"

void demo(cl_context context, cl_device_id device) {

    // Draw Rect
    draw_rect_demo(context, device);

    // Generate Gradient Color Image
    gradient_image_demo(context, device);

    // Rotate
    rotate_demo(context, device);

    // Crop
    crop_demo(context, device);

    // Convolution
    conv_demo(context, device);

    // Resize demo
    resize_demo(context, device);

    // Convert Channel demo
    convert_channel_demo(context, device);

    // Merge two images demo
    merge_demo(context, device);

    convert_gray_demo(context, device);

    // Image Mirror Horizontal
    mirror_demo(context, device);

    // Mask demo
    mask_video_demo(context, device);

}

int main() {

#ifndef DEBUG_MODE
    KHM::sayHello();
#endif

    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    // All features demo
    demo(context, device);

    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}