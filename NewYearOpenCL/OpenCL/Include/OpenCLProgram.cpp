// OpenCL Auto Build Kernel Source and Release Memory
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include <cstring>

#include "OpenCLProgram.h"
#include "OpenCLError.h"

OpenCLProgram::OpenCLProgram(
        cl_context context, cl_device_id device, const char *kernel_name,
        const char *cl_kernel_source_code
) :
        program_kernel_name(strdup(kernel_name)) {
//    std::cout << "Building " << kernel_name << "..." << std::endl;
    program = CLCreateProgram(context, device, cl_kernel_source_code);
}

cl_kernel OpenCLProgram::CreateKernel() {
    cl_int err;
    cl_kernel kernel =
            clCreateKernel(
                    program,
                    program_kernel_name,
                    &err
            );
    CHECK_CL_ERROR(err, "Failed to create kernel.");
    return kernel;
}

void OpenCLProgram::ReleaseProgram() {
    if (!isReleased) {
        isReleased = true;
        clReleaseProgram(program);
    }
}

OpenCLProgram::~OpenCLProgram() {
//    std::cout << "Class OpenCLProgram " << program_kernel_name << " Destructor called" << std::endl;
    ReleaseProgram();
    free(program_kernel_name);
}