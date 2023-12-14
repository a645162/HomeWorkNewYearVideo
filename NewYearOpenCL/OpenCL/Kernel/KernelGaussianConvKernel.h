//
// Created by konghaomin on 23-12-13.
//

#ifndef NEW_YEAR_OPENCL_KERNEL_CONVOLUTION_H
#define NEW_YEAR_OPENCL_KERNEL_CONVOLUTION_H

const char *cl_kernel_gaussian_conv_kernel = R"(// GaussianConvKernel.cl
// OpenCL kernel for generating Gaussian convolution kernels on GPU

__kernel void matrixElementSum(__global float *matrix, __global float *result,
                               int size) {
    size_t tid = get_local_id(0) + get_local_id(1) * get_local_size(0);
    size_t stride = get_local_size(0) * get_local_size(1);

    // Perform block-wise reduction
    for (size_t i = tid; i < size * size; i += stride) {
        atomic_add(result, matrix[i]);
    }
}

__kernel void matrixElementWiseDivision(__global float *matrix,
                                        const __global float *divisor,
                                        int size) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if (x < size && y < size) {
        matrix[y * size + x] /= *divisor;
    }
}

// Define OpenCL kernel for Gaussian kernel generation
__kernel void generateGaussianKernel(__global float *kernel, int size,
                                     float strength) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if (x < size && y < size) {
        float center = (float)(size - 1) / 2;

        float value =
            (float)(1.0f / (2.0f * M_PI * pow(strength, 2))) *
            exp(-(pow((float)x - center, 2) + pow((float)y - center, 2)) /
                (2 * strength * strength));

        kernel[y * size + x] =
            value; // Divide by strength to match Python implementation
    }
}
)";

#endif //NEW_YEAR_OPENCL_KERNEL_CONVOLUTION_H
