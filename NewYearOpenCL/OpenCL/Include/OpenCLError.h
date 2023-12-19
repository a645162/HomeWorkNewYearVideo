// OpenCL Error Handle
// Created by Haomin Kong on 2023/12/13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_OPENCL_ERROR_H
#define NEW_YEAR_OPENCL_OPENCL_ERROR_H

#include <iostream>
#include "OpenCLInclude.h"

#define CHECK_CL_ERROR(err, msg) \
    {if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL error (" << err << "): " << clGetErrorString(err) << " - " << msg << std::endl; \
        exit(EXIT_FAILURE); \
    }}

const char* clGetErrorString(cl_int error);

#endif //NEW_YEAR_OPENCL_OPENCL_ERROR_H
