// OpenCL Kernel RAII
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include "OpenCLKernel.h"

#include "../OpenCLError.h"

#include "../OpenCLWorkFlow.h"

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

cl_kernel OpenCLKernel::GetKernel() const {
    if (isReleased()) {
        std::cerr << "Error: OpenCL Kernel is released." << std::endl;
        return nullptr;
    }

    return kernel;
}

void OpenCLKernel::Execute(
    cl_command_queue queue,
    size_t work_dim,
    size_t* global_work_size,
    const bool wait_finish
) const {
    if (isReleased()) {
        std::cerr << "Error: OpenCL Kernel is released." << std::endl;
        return;
    }

    CLKernelEnqueue(queue, kernel, work_dim, global_work_size);

    if (wait_finish) {
        clFinish(queue);
    }
}

bool OpenCLKernel::isReleased() const {
    return isPtrReleased;
}

void OpenCLKernel::ReleaseKernel() {
    if (!isReleased()) {
        isPtrReleased = true;
        clReleaseKernel(kernel);
    }
}

OpenCLKernel::~OpenCLKernel() {
    ReleaseKernel();
}
