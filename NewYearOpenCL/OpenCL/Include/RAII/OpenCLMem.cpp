// OpenCL Memory RAII
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include "OpenCLMem.h"

#include "../OpenCLError.h"

OpenCLMem::OpenCLMem(
    cl_context context,
    size_t size,
    cl_mem_flags flags,
    void* host_ptr
) {
    mem = OpenCLMalloc(context, size, flags, host_ptr);
}

cl_mem OpenCLMem::GetMem() const {
    return mem;
}

bool OpenCLMem::isReleased() const {
    return isPtrReleased;
}

void OpenCLMem::Release() {
    if (!isReleased()) {
        const cl_int err = clReleaseMemObject(mem);
        CHECK_CL_ERROR(err, "clReleaseMemObject");
        isPtrReleased = true;
    }
}

OpenCLMem::~OpenCLMem() {
    Release();
}
