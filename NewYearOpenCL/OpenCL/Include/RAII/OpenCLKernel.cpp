// OpenCL Auto Release Kernel Memory
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include "OpenCLKernel.h"

#include "../OpenCLError.h"

OpenCLKernel::OpenCLKernel(cl_program program, const char* kernel_name) {
    cl_int err;
    kernel =
            clCreateKernel(
                program,
                kernel_name,
                &err
            );
    CHECK_CL_ERROR(err, "Failed to create kernel.");
}

void OpenCLKernel::ReleaseKernel() {
    if (!isReleased) {
        isReleased = true;
        clReleaseKernel(kernel);
    }
}

OpenCLKernel::~OpenCLKernel() {
    ReleaseKernel();
}
