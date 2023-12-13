//
// Created by konghaomin on 23-12-13.
//

#ifndef NEWYEAROPENCL_OPENCLPROGRAM_H
#define NEWYEAROPENCL_OPENCLPROGRAM_H

#include "OpenCLInclude.h"
#include "OpenCLFlow.h"

class OpenCLProgram {

private:
    cl_program program;
    char *program_kernel_name;
public:
    OpenCLProgram(cl_context context,
                  cl_device_id device,
                  const char *kernel_name,
                  const char *cl_kernel_source_code
    );

    cl_kernel CreateKernel();

    ~OpenCLProgram();
};

#endif //NEWYEAROPENCL_OPENCLPROGRAM_H
