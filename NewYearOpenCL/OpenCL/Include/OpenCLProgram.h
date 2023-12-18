// OpenCL Auto Build Kernel Source and Release Memory
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#ifndef NEWYEAROPENCL_OPENCLPROGRAM_H
#define NEWYEAROPENCL_OPENCLPROGRAM_H

#include "OpenCLInclude.h"
#include "OpenCLFlow.h"

class OpenCLProgram {

private:
    cl_program program;
    char *program_kernel_name;
    bool isReleased = false;
public:
    OpenCLProgram(
            cl_context context,
            cl_device_id device,
            const char *kernel_name,
            const char *cl_kernel_source_code
    );

    cl_kernel CreateKernel();

    void ReleaseProgram();

    ~OpenCLProgram();
};

#endif //NEWYEAROPENCL_OPENCLPROGRAM_H
