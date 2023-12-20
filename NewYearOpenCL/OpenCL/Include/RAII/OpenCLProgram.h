// OpenCL Auto Build Kernel Source and Release Memory
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo
// RAII (Resource Acquisition Is Initialization)

#ifndef OPENCL_OPENCL_PROGRAM_H
#define OPENCL_OPENCL_PROGRAM_H

#include "../OpenCLInclude.h"
#include "../OpenCLWorkFlow.h"

#include "OpenCLKernel.h"

class OpenCLProgram {
private:
	cl_program program;
	char* program_kernel_name;
	bool isPtrReleased = false;

public:
	OpenCLProgram(
		cl_context context,
		cl_device_id device,
		const char* kernel_name,
		const char* cl_kernel_source_code
	);

	[[nodiscard]] cl_kernel CreateKernel() const;

	OpenCLKernel CreateKernelRAII();

	[[nodiscard]] bool isReleased() const;

	void ReleaseProgram();

	~OpenCLProgram();
};

#endif //OPENCL_OPENCL_PROGRAM_H
