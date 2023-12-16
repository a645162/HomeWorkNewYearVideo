// Demo:Image Operation on GPU
// Created by Haomin Kong on 23-12-16.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "../Devices/OpenCLDevices.h"
#include "../Include/OpenCLInclude.h"

#include "Draw/DrawRect/DrawRectDemo.h"

#include "Generate/GradientImage/GradientImageDemo.h"

#include "Image/Convolution/ImageConvolutionDemo.h"
#include "Image/Convolution/ImageGaussianBlurDemo.h"

int main(){

    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    // All features demo

    // Generate Gradient Color Image
    gradient_image_demo(context, device);

    // Draw Rect
//    draw_rect_demo(context, device);

//    blur_conv_demo(context, device);

//    conv_demo(context, device);

    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}