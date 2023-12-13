//
// Created by 孔昊旻 on 2023/12/13.
//

#ifndef NEW_YEAR_OPENCL_DEBUG_VAR_H
#define NEW_YEAR_OPENCL_DEBUG_VAR_H

#endif //NEW_YEAR_OPENCL_DEBUG_VAR_H

#ifdef DEBUG_MODE

#define USE_DEBUG_CL_PLATFORM_DEVICE_INDEX

#ifdef __APPLE__

#define DEBUG_CL_PLATFORM_INDEX 0
#define DEBUG_CL_DEVICE_INDEX 2

#else

#ifdef _WIN32
// 32-bit Windows code
#endif

#ifdef _WIN64
// 64-bit Windows code

// AMD Vega 64
#define DEBUG_CL_PLATFORM_INDEX 0
// NVIDIA Tesla P40
//#define DEBUG_CL_PLATFORM_INDEX 1
// E5 2690 V4
//#define DEBUG_CL_PLATFORM_INDEX 2

#define DEBUG_CL_DEVICE_INDEX 0

#endif

#ifdef __linux__

// E5 2690 V4
//#define DEBUG_CL_PLATFORM_INDEX 0
// NVIDIA Tesla P40
#define DEBUG_CL_PLATFORM_INDEX 1
// AMD Vega 64
//#define DEBUG_CL_PLATFORM_INDEX 2

#define DEBUG_CL_DEVICE_INDEX 0

#endif

#endif

#endif