// OpenCL Kernel Source Build Check
// Created by Haomin Kong on 23-12-20.
// https://github.com/a645162/HomeWorkNewYearVideo

#include <iostream>

#include "../../OpenCL/Include/OpenCLWorkFlow.h"

#include "../../OpenCL/Include/OpenCLRAII.h"

#include "../../OpenCL/Devices/OpenCLDevices.h"

#include "../../OpenCL/Image/Draw/DrawRect.h"

#include "../../OpenCL/Image/Generate/GenerateGradientImage.h"

#include "../../OpenCL/Image/ImageConvolution.h"
#include "../../OpenCL/Image/ImageCrop.h"
#include "../../OpenCL/Image/ImageMask.h"
#include "../../OpenCL/Image/ImageMerge.h"
#include "../../OpenCL/Image/ImageMirror.h"
#include "../../OpenCL/Image/ImageResize.h"
#include "../../OpenCL/Image/ImageRotate.h"
#include "../../OpenCL/Image/ImageChannelConvert.h"
#include "../../OpenCL/Image/ImageGrayRGB.h"
#include "../../OpenCL/Image/ImageReverseColor.h"

#include "../../OpenCL/Utils/OpenCLMemset.h"

int main()
{
    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    CLCreateProgram_Draw_Rect(context, device);

    CLCreateProgram_Generate_GradientColor(context, device);
    CLCreateProgram_Generate_GradientImage(context, device);

    CLCreateProgram_Image_Conv(context, device);
    CLCreateProgram_Image_Crop(context, device);
    CLCreateProgram_Image_Mask(context, device);
    CLCreateProgram_Image_Merge(context, device);
    CLCreateProgram_Image_Mirror(context, device);
    CLCreateProgram_Image_Resize(context, device);
    CLCreateProgram_Image_Rotate(context, device);
    CLCreateProgram_Image_Channel(context, device);
    CLCreateProgram_Image_Gray_RGB(context, device);
    CLCreateProgram_Image_Reverse_Color(context, device);

    CLCreateProgram_Memset_2D(context, device);

    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}
