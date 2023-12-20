// OpenCL Queue RAII
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include "OpenCLQueue.h"

#include "../OpenCLError.h"

OpenCLQueue::OpenCLQueue(cl_context context, cl_device_id device) {
    queue = CLCreateCommandQueue(context, device);
}

cl_command_queue OpenCLQueue::GetQueue() const {
    if (isReleased()) {
        std::cerr << "Error: OpenCL Queue is released." << std::endl;
        return nullptr;
    }

    return queue;
}

void OpenCLQueue::KernelEnqueue(
    cl_kernel kernel,
    size_t work_dim,
    size_t* global_work_size,
    const bool wait_finish
) const {
    if (isReleased()) {
        std::cerr << "Error: OpenCL Queue is released." << std::endl;
        return;
    }

    CLKernelEnqueue(queue, kernel, work_dim, global_work_size);

    if (wait_finish) {
        clFinish(queue);
    }
}

void OpenCLQueue::WaitFinish() const {
    if (isReleased()) {
        std::cerr << "Error: OpenCL Queue is released." << std::endl;
        return;
    }

    clFinish(queue);
}

bool OpenCLQueue::isReleased() const {
    return isPtrReleased;
}

void OpenCLQueue::Release() {
    if (!isReleased()) {
        isPtrReleased = true;
        clReleaseCommandQueue(queue);
    }
}

OpenCLQueue::~OpenCLQueue() {
    Release();
}
