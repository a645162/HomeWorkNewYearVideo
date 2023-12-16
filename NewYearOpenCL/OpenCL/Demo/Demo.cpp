//
// Created by konghaomin on 23-12-16.
//

#include "../Devices/OpenCLDevices.h"
#include "../Include/OpenCLInclude.h"

#include "ImageConvolution/ImageConvolutionDemo.h"

int main(){

    cl_device_id device = UserSelectDevice();
    cl_context context = CLCreateContext(device);

    // All features demo
    conv_demo(context, device);

    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}