// OpenCL Work Flow
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo
// OpenCL Flow Simplified and add error checking.

#include "OpenCLWorkFlow.h"
#include "OpenCLError.h"

#include <vector>

#include "../../Utils/Calc.h"

cl_context CLCreateContext(cl_device_id device) {
    // Create OpenCL context
    cl_int err;
    cl_context context =
            clCreateContext(
                nullptr, 1, &device,
                nullptr, nullptr, &err
            );
    CHECK_CL_ERROR(err, "Failed to create context.");
    return context;
}

cl_command_queue CLCreateCommandQueue(cl_context context, cl_device_id device) {
    // Create OpenCL command queue
    cl_int err;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERROR(err, "Failed to create command queue.");
    return queue;
}

cl_program CLCreateProgram(cl_context context, cl_device_id device, const char* cl_kernel_source_code) {
    // Create and build OpenCL program
    cl_program program =
            clCreateProgramWithSource(
                context,
                1,
                &cl_kernel_source_code,
                nullptr,
                nullptr
            );

    auto err_build = clBuildProgram(
        program, 1, &device, nullptr,
        nullptr, nullptr
    );

    if (err_build != CL_SUCCESS) {
        size_t log_size;
        int err;
        err = clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG,
            0, nullptr, &log_size
        );
        CHECK_CL_ERROR(err, "clGetProgramBuildInfo");

        std::vector<char> build_log(log_size);
        err = clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG,
            log_size, build_log.data(), nullptr
        );
        CHECK_CL_ERROR(err, "clGetProgramBuildInfo");

        std::cerr << "Build Error!\n\tlog:\n" << build_log.data() << std::endl;
        std::cerr << "Kernel Source Code:" << cl_kernel_source_code << std::endl;

        std::cerr << "OpenCL error (" << err_build << "): " << clGetErrorString(err_build) << std::endl;
        exit(EXIT_FAILURE);
    }

    //    CHECK_CL_ERROR(
    //            err,
    //            "Failed to build program."
    //    );

    return program;
}

cl_mem OpenCLMalloc(cl_context context, size_t size, cl_mem_flags flags, void* host_ptr) {
    // Allocate OpenCL memory
    cl_int err;
    cl_mem mem = clCreateBuffer(context, flags, size, host_ptr, &err);
    CHECK_CL_ERROR(err, "Failed to create buffer.");
    return mem;
}

void OpenCLMemcpyFromDevice(
    cl_command_queue queue,
    void* dst_cpu,
    cl_mem src_device,
    size_t size
) {
    cl_int err;
    err = clEnqueueReadBuffer(
        queue,
        src_device,
        CL_TRUE,
        0,
        size,
        dst_cpu,
        0,
        nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "Failed to read buffer.");
}

void OpenCLMemcpyFromDevice(
    cl_command_queue queue,
    void* dst_cpu,
    cl_mem src_device,
    int width, int height, int channel
) {
    OpenCLMemcpyFromDevice(queue, dst_cpu, src_device, calcImageSize(width, height, channel));
}

unsigned int OpenCLSetKernelArg(
    cl_kernel kernel,
    cl_uint* index_var,
    size_t size_of_type,
    const void* value
) {
    cl_int err;
    err = clSetKernelArg(kernel, (*index_var)++, size_of_type, value);
    CHECK_CL_ERROR(err, "Failed to set kernel argument.");
    return (*index_var);
}

void CLKernelEnqueue(
    cl_command_queue queue,
    cl_kernel kernel,
    size_t work_dim,
    size_t* global_work_size
) {
    cl_int err;
    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        static_cast<cl_uint>(work_dim),
        nullptr,
        global_work_size,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "Failed to enqueue kernel.");
}
