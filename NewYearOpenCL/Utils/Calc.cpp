// Calc Some Values
// Created by Haomin Kong on 2023/12/18.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "Calc.h"

#include "../OpenCL/Include/OpenCLInclude.h"

size_t calcGlobalWorkSize(size_t localWorkSize, size_t globalWorkSize) {
    size_t r = globalWorkSize % localWorkSize;
    if (r == 0) {
        return globalWorkSize;
    } else {
        return globalWorkSize + localWorkSize - r;
    }
}

size_t calcLocalWorkSize(size_t localWorkSize, size_t globalWorkSize) {
    size_t r = globalWorkSize % localWorkSize;
    if (r == 0) {
        return localWorkSize;
    } else {
        return globalWorkSize / (globalWorkSize / localWorkSize + 1);
    }
}

unsigned int calcImageSize(
        unsigned int width, unsigned int height, unsigned int channels
) {
    return width * height * channels * sizeof(unsigned char);
}