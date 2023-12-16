//
// Created by konghaomin on 23-12-13.
//

#ifndef NEWYEAROPENCL_IMAGEGAUSSIANBLUR_H
#define NEWYEAROPENCL_IMAGEGAUSSIANBLUR_H

typedef struct {
    int size;
    float strength;
} GaussianKernelParameters;

float *createGaussianKernel(int size, float strength);

float *createGaussianKernel(GaussianKernelParameters parameters);

GaussianKernelParameters getGaussianKernelParameters(int size, float strength);

GaussianKernelParameters calcGaussianKernelParameters(float blur_intensity);

#endif //NEWYEAROPENCL_IMAGEGAUSSIANBLUR_H
