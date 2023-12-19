// OpenCL Memory RAII
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include "OpenCLMem.h"

#include "../OpenCLError.h"
#include "../../../Utils/Calc.h"

OpenCLMem::OpenCLMem(
    cl_context context,
    size_t size,
    cl_mem_flags flags,
    void* host_ptr
): mem_size(size) {
    mem = OpenCLMalloc(context, size, flags, host_ptr);
}

OpenCLMem::OpenCLMem(
    cl_context context,
    unsigned int width,
    unsigned int height,
    unsigned int channel,
    cl_mem_flags flags,
    void* host_ptr
) {
    mem_size = calcImageSize(width, height, channel);
    mem = OpenCLMalloc(
        context,
        mem_size,
        flags,
        host_ptr
    );
}

cl_mem OpenCLMem::GetMem() const {
    return mem;
}

void OpenCLMem::CopyToHost(cl_command_queue queue, void* dst_cpu) const {
    if (!isReleased()) {
        OpenCLMemcpyFromDevice(
            queue,
            dst_cpu,
            mem,
            mem_size
        );
    }
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
