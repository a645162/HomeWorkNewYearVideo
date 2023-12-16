//
// Created by konghaomin on 23-12-16.
//

#ifndef NEWYEAROPENCL_IMAGEGAUSSIANBLURDEMO_H
#define NEWYEAROPENCL_IMAGEGAUSSIANBLURDEMO_H

#include "../../../../OpenCV/Include/OpenCVInclude.h"

#include "../../../Include/OpenCLInclude.h"
#include "../../../Include/OpenCLError.h"
#include "../../../Include/OpenCLFlow.h"
#include "../../../Include/OpenCLProgram.h"

void blur_conv_demo(cl_context context, cl_device_id device);

#endif //NEWYEAROPENCL_IMAGEGAUSSIANBLURDEMO_H
