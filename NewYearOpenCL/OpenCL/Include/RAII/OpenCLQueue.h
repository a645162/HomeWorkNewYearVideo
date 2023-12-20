// OpenCL Queue RAII
// Created by Haomin Kong on 2023/12/19.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#ifndef OPENCL_OPENCL_QUEUE_H
#define OPENCL_OPENCL_QUEUE_H

#include "../OpenCLInclude.h"
#include "../OpenCLWorkFlow.h"

class OpenCLQueue {
private:
    cl_command_queue queue;
    bool isPtrReleased = false;

public:
    OpenCLQueue(cl_context context, cl_device_id device);

    [[nodiscard]] cl_command_queue GetQueue() const;

    void KernelEnqueue(
        cl_kernel kernel,
        size_t work_dim,
        size_t* global_work_size,
        bool wait_finish = true
    ) const;

    void WaitFinish() const;

    [[nodiscard]] bool isReleased() const;

    void Release();

    ~OpenCLQueue();
};

#endif //OPENCL_OPENCL_QUEUE_H
