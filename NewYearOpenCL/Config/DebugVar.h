// Some Debug Variable
// Created by Haomin Kong on 2023/12/13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_DEBUG_VAR_H
#define NEW_YEAR_OPENCL_DEBUG_VAR_H

#endif //NEW_YEAR_OPENCL_DEBUG_VAR_H

#ifdef DEBUG_MODE

#define USE_DEBUG_CL_PLATFORM_DEVICE_INDEX

#ifdef __APPLE__

#define DEBUG_CL_PLATFORM_INDEX 0

// Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
//#define DEBUG_CL_DEVICE_INDEX 0
// Intel(R) UHD Graphics 630
#define DEBUG_CL_DEVICE_INDEX 1
// AMD Radeon Pro 560X Compute Engine
//#define DEBUG_CL_DEVICE_INDEX 2

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