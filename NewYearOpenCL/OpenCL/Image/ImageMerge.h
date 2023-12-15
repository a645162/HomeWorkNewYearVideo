//
// Created by 孔昊旻 on 2023/12/15.
//

#ifndef NEWYEAROPENCL_IMAGEMERGE_H
#define NEWYEAROPENCL_IMAGEMERGE_H

#include "../Include/OpenCLInclude.h"
#include "../Include/OpenCLError.h"
#include "../Include/OpenCLFlow.h"
#include "../Include/OpenCLProgram.h"

OpenCLProgram CLCreateProgramImageMerge(cl_context context, cl_device_id device);

void KernelSetArgImageMerge(
        cl_kernel kernel,
        cl_mem image1,
        cl_mem image2,
        cl_mem device_output,
        int image1_width, int image1_height, int image1_channels,
        int image2_target_x, int image2_target_y,
        int image2_width, int image2_height, int image2_channels,
        int image2_alpha
);

void merge_demo(cl_context context, cl_device_id device);

#endif //NEWYEAROPENCL_IMAGEMERGE_H
