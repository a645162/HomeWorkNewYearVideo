// OpenCL Memset
// Created by Haomin Kong on 23-12-19.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef OPENCL_MEMSET_H
#define OPENCL_MEMSET_H

#include "../Include/OpenCLInclude.h"

#include "../Include/OpenCLRAII.h"

OpenCLProgram CLCreateProgram_Memset_2D(cl_context context, cl_device_id device);

void KernelSetArg_Memset_2D(
    cl_kernel kernel,
    cl_mem device_target,
    int width, int height, int channel,
    unsigned char value
);

#endif //OPENCL_MEMSET_H
