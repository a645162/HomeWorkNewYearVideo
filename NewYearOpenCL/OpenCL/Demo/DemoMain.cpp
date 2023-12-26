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
#include "Image/Mask/ImageMaskDemo.h"
#include "Image/Mask/MaskAndChannelDemo.h"
#include "Image/ReverseColor/ImageReverseColorDemo.h"

void demo(cl_context context, cl_device_id device, int index) {
    switch (index) {
        case 1:
            // Image Mirror Horizontal
            mirror_demo(context, device);
            break;
        case 2:
            // Merge two images demo
            merge_demo(context, device);
            break;
        case 3:
            // Convert Channel demo
            convert_channel_demo(context, device);
            break;
        case 4:
            // Convert to Gray
            convert_gray_demo(context, device);
            break;
        case 5:
            // Resize demo
            resize_demo(context, device);
            break;
        case 6:
            // Crop
            crop_demo(context, device);
            break;
        case 7:
            // Rotate
            rotate_demo(context, device);
            break;
        case 8:
            // Generate Gradient Color Image
            gradient_image_demo(context, device);
            break;
        case 9:
            // Draw Rect
            draw_rect_demo(context, device);
            break;
        case 10:
            // Convolution and Binaryzation
            conv_demo(context, device);
            break;
        case 11:
            // Gaussian Blur Convolution
            blur_conv_demo(context, device);
            break;
        case 12:
            // Mask demo
            mask_video_demo(context, device);
            break;
        case 13:
            // Mask and Channel demo
            mask_channel_demo(context, device);
            break;
        case 14:
            // Image Reverse Color demo
            reverse_color_demo(context, device);
            break;
        default:
            std::cout << "Index Invaild!" << std::endl;
            break;
    }
}

int main() {
    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    // All features demo
    int index = 0;

    std::cout << "Please input the index of demo: " << std::endl;
    std::cout << "1. Image Mirror Horizontal" << std::endl;
    std::cout << "2. Merge two images demo" << std::endl;
    std::cout << "3. Convert Channel demo" << std::endl;
    std::cout << "4. Convert to Gray" << std::endl;
    std::cout << "5. Resize demo" << std::endl;
    std::cout << "6. Crop" << std::endl;
    std::cout << "7. Rotate" << std::endl;
    std::cout << "8. Generate Gradient Color Image" << std::endl;
    std::cout << "9. Draw Rect" << std::endl;
    std::cout << "10. Convolution then Binaryzation" << std::endl;
    std::cout << "11. Gaussian Blur Convolution" << std::endl;
    std::cout << "12. Mask demo" << std::endl;
    std::cout << "13. Mask and Channel demo" << std::endl;
    std::cout << "14. Reverse Color demo" << std::endl;

    std::cin >> index;

    if (index == 0) {
        for (int i = 1; i < 14 + 1; ++i) {
            demo(context, device, i);
        }
    }

    demo(context, device, index);

    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}
