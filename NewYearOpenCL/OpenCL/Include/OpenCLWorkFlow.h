// OpenCL Work Flow
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo
// OpenCL Flow Simplified and add error checking.

#ifndef OPENCL_OPENCL_WORK_FLOW_H
#define OPENCL_OPENCL_WORK_FLOW_H

#include "OpenCLInclude.h"

cl_program CLCreateProgram(cl_context context, cl_device_id device, const char* cl_kernel_source_code);

cl_context CLCreateContext(cl_device_id device);

cl_command_queue CLCreateCommandQueue(cl_context context, cl_device_id device);

cl_mem OpenCLMalloc(cl_context context, size_t size, cl_mem_flags flags, void* host_ptr);

void OpenCLMemcpyFromDevice(
    cl_command_queue queue,
    void* dst_cpu,
    cl_mem src_device,
    size_t size
);

unsigned int OpenCLSetKernelArg(
    cl_kernel kernel,
    cl_uint* index_var,
    size_t size_of_type,
    const void* value
);

void CLKernelEnqueue(
    cl_command_queue queue,
    cl_kernel kernel,
    size_t work_dim,
    size_t* global_work_size
);

#endif //OPENCL_OPENCL_WORK_FLOW_H
