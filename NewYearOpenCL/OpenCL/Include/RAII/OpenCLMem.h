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
    size_t mem_size;
    bool isPtrReleased = false;

public:
    OpenCLMem(
        cl_context context,
        size_t size,
        cl_mem_flags flags = CL_MEM_READ_WRITE,
        void* host_ptr = nullptr
    );

    OpenCLMem(
        cl_context context,
        unsigned int width,
        unsigned int height,
        unsigned int channel,
        cl_mem_flags flags = CL_MEM_READ_WRITE,
        void* host_ptr = nullptr
    );

    [[nodiscard]] cl_mem GetMem() const;

    operator cl_mem() const {
        return GetMem();
    }

    void CopyToHost(cl_command_queue queue, void* dst_cpu) const;

    [[nodiscard]] bool isReleased() const;

    void Release();

    ~OpenCLMem();
};

OpenCLMem OpenCLMemFromHost(
    cl_context context,
    unsigned int width,
    unsigned int height,
    unsigned int channel,
    void* host_ptr,
    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
);

#endif //OPENCL_OPENCL_MEM_H
