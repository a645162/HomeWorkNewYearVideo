// OpenCL Auto Release Kernel Memory
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#ifndef OPENCL_KERNEL_H
#define OPENCL_KERNEL_H

#include "../OpenCLInclude.h"

class OpenCLKernel {
private:
    bool isReleased = false;

public:
    cl_kernel kernel;

    OpenCLKernel(
        cl_program program,
        const char* kernel_name
    );

    void ReleaseKernel();

    ~OpenCLKernel();
};


#endif //OPENCL_KERNEL_H
