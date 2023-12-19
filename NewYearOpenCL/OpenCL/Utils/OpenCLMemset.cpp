// OpenCL Memset
// Created by Haomin Kong on 23-12-19.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "OpenCLMemset.h"

#include "../Include/OpenCLWorkFlow.h"

#include "../Kernel/KernelOpenCLMemset2D.h"

OpenCLProgram CLCreateProgram_Memset_2D(cl_context context, cl_device_id device) {
    return {
        context,
        device,
        "OpenCLMemset2D",
        cl_kernel_opencl_memset_2d
    };
}

void KernelSetArg_Memset_2D(
    cl_kernel kernel,
    cl_mem device_target,
    int width, int height, int channel,
    unsigned char value
) {
    cl_uint kernel_arg_index1 = 0;

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(cl_mem), &device_target);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &width);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &height);
    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(int), &channel);

    OpenCLSetKernelArg(kernel, &kernel_arg_index1, sizeof(unsigned char), &value);
}
