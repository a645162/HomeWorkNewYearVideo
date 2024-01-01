// OpenCL Memory RAII
// Created by Haomin Kong on 2023/12/15.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#include "OpenCLMem.h"

#include "../../../OpenCV/Include/OpenCVInclude.h"

#include "../OpenCLError.h"
#include "../../../Utils/Calc.h"

OpenCLMem::OpenCLMem(
    cl_context context,
    size_t size,
    cl_mem_flags flags,
    void* host_ptr
): mem_size(size)
{
    mem = OpenCLMalloc(context, size, flags, host_ptr);
}

OpenCLMem::OpenCLMem(
    cl_context context,
    unsigned int width,
    unsigned int height,
    unsigned int channel,
    cl_mem_flags flags,
    void* host_ptr
) : OpenCLMem(context, calcImageSize(width, height, channel), flags, host_ptr)
{
    this->width = width;
    this->height = height;
    this->channel = channel;
}

bool OpenCLMem::isSizeVaild() const
{
    return width > 0 && height > 0 && channel > 0;
}

cl_mem OpenCLMem::GetMem() const
{
    return mem;
}

void OpenCLMem::CopyToHost(cl_command_queue queue, void* dst_cpu) const
{
    if (!isReleased())
    {
        OpenCLMemcpyFromDevice(
            queue,
            dst_cpu,
            mem,
            mem_size
        );
    }
}

void OpenCLMem::CopyFromOtherMem(cl_command_queue queue, cl_mem src) const
{
    const auto err = clEnqueueCopyBuffer(
        queue,
        src,
        mem,
        0,
        0,
        mem_size,
        0,
        nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "clEnqueueCopyBuffer(FromOtherMem)");
}

void OpenCLMem::CopyToOtherMem(cl_command_queue queue, cl_mem dst) const
{
    const auto err = clEnqueueCopyBuffer(
        queue,
        mem,
        dst,
        0,
        0,
        mem_size,
        0,
        nullptr,
        nullptr
    );
    CHECK_CL_ERROR(err, "clEnqueueCopyBuffer(CopyToOtherMem)");
}

void OpenCLMem::ShowByOpenCV(
    cl_command_queue queue,
    int width, int height, int channel,
    int wait_time
) const
{
    if (!isReleased())
    {
        cv::Mat mat(height, width, CV_MAKETYPE(CV_8U, channel));
        CopyToHost(queue, mat.data);
        std::cout << "Image " << width << "x" << height << "x" << channel << std::endl;
        cv::imshow("Image", mat);
        cv::waitKey(wait_time);
        mat.release();
    }
}

void OpenCLMem::ShowByOpenCV(
    cl_command_queue queue, int wait_time
) const
{
#ifndef DEBUG_MODE
    return;
#endif

    if (isSizeVaild())
    {
        ShowByOpenCV(
            queue,
            static_cast<int>(this->width),
            static_cast<int>(this->height),
            static_cast<int>(this->channel),
            wait_time
        );
    }
    else
    {
        std::cout << "This Memory is not initlize by size!" << std::endl;
    }
}

bool OpenCLMem::isReleased() const
{
    return isPtrReleased;
}

void OpenCLMem::Release()
{
    if (!isReleased())
    {
        const cl_int err = clReleaseMemObject(mem);
        CHECK_CL_ERROR(err, "clReleaseMemObject");
        isPtrReleased = true;
    }
}

OpenCLMem OpenCLMemFromHost(
    cl_context context,
    unsigned int width,
    unsigned int height,
    unsigned int channel,
    void* host_ptr,
    cl_mem_flags flags
)
{
    return {context, width, height, channel, flags, host_ptr};
}

OpenCLMem::~OpenCLMem()
{
    Release();
}
