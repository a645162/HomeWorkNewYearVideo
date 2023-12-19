// OpenCL Memory RAII
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#ifndef OPENCL_OPENCL_MEM_H
#define OPENCL_OPENCL_MEM_H

#include "../OpenCLInclude.h"
#include "../OpenCLWorkFlow.h"

class OpenCLMem {
private:
    cl_mem mem;
    bool isPtrReleased = false;

public:
    OpenCLMem(
        cl_context context,
        size_t size,
        cl_mem_flags flags,
        void* host_ptr
    );

    [[nodiscard]] cl_mem GetMem() const;

    [[nodiscard]] bool isReleased() const;

    void Release();

    ~OpenCLMem();
};

#endif //OPENCL_OPENCL_MEM_H
