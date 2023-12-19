// Generate Gaussian Convolution Kernel
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#ifndef NEW_YEAR_OPENCL_GAUSSIAN_KERNEL_H
#define NEW_YEAR_OPENCL_GAUSSIAN_KERNEL_H

typedef struct {
    int size;
    float strength;
} GaussianKernelParameters;

float* createGaussianKernel(int size, float strength);

float* createGaussianKernel(GaussianKernelParameters parameters);

GaussianKernelParameters getGaussianKernelParameters(int size, float strength);

GaussianKernelParameters calcGaussianKernelParameters(float blur_intensity);

#endif //NEW_YEAR_OPENCL_GAUSSIAN_KERNEL_H
