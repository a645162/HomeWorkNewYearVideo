//
// Created by konghaomin on 2023/12/9.
//

#ifndef CUDA_INCLUDE_CUH
#define CUDA_INCLUDE_CUH

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#define GET_INDEX_ON_CUDA(x, y, width, channels) (((y) * (width) + (x)) * (channels))

#endif //CUDA_INCLUDE_CUH
