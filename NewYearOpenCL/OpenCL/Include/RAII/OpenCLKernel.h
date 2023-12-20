// OpenCL Kernel RAII
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#ifndef OPENCL_KERNEL_H
#define OPENCL_KERNEL_H

#include "../OpenCLInclude.h"

class OpenCLKernel {
private:
    bool isPtrReleased = false;
    cl_kernel kernel;

public:
    OpenCLKernel(
        cl_program program,
        const char* kernel_name
    );

    [[nodiscard]] cl_kernel GetKernel() const;

    void KernelEnqueue(
        cl_command_queue queue,
        size_t work_dim,
        size_t* global_work_size,
        bool wait_finish = true
    ) const;

    [[nodiscard]] bool isReleased() const;

    void ReleaseKernel();

    ~OpenCLKernel();
};


#endif //OPENCL_KERNEL_H
