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
    return queue;
}

void OpenCLQueue::WaitFinish() const {
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
