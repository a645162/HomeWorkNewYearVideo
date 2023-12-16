//
// Created by konghaomin on 23-12-13.
//

#ifndef NEWYEAROPENCL_IMAGECONVOLUTION_H
#define NEWYEAROPENCL_IMAGECONVOLUTION_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLFlow.h"
#include "../Include/OpenCLProgram.h"

OpenCLProgram CLCreateProgram_Image_Conv(cl_context context, cl_device_id device);

void KernelSetArg_Image_Conv(
        cl_kernel kernel,
        cl_mem device_src,
        cl_mem device_dst,
        int height,
        int width,
        int channels,
        cl_mem conv_kernel,
        int conv_kernel_size,
        int padSize
);

#endif //NEWYEAROPENCL_IMAGECONVOLUTION_H
