// Demo:Image Operation on GPU
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "../Devices/OpenCLDevices.h"
#include "../Include/OpenCLInclude.h"

#include "Draw/DrawRect/DrawRectDemo.h"

#include "Generate/GradientImage/GradientImageDemo.h"

#include "Image/Convolution/ImageConvolutionDemo.h"
#include "Image/Convolution/ImageGaussianBlurDemo.h"
#include "Image/Rotate/ImageRotateDemo.h"
#include "Image/Crop/ImageCropDemo.h"
#include "Image/Resize/ImageResizeDemo.h"
#include "Image/Channel/ImageChannelDemo.h"
#include "Image/Channel/ImageGrayRGBDemo.h"
#include "Image/Merge/ImageMergeDemo.h"
#include "Image/Mirror/ImageMirrorDemo.h"

void demo(cl_context context, cl_device_id device) {

    // Merge two images demo
    merge_demo(context, device);

    // Convert to Gray
    convert_gray_demo(context, device);

    // Convert Channel demo
    convert_channel_demo(context, device);

    // Resize demo
    resize_demo(context, device);

    // Crop
    crop_demo(context, device);

    // Rotate
    rotate_demo(context, device);

    // Generate Gradient Color Image
    gradient_image_demo(context, device);

    // Draw Rect
    draw_rect_demo(context, device);

    // Convolution
    conv_demo(context, device);

    // Gaussian Blur Convolution
    blur_conv_demo(context, device);

}

int main() {

    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    // All features demo
//    demo(context, device);

    // Image Mirror Horizontal
    mirror_demo(context, device);

    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}