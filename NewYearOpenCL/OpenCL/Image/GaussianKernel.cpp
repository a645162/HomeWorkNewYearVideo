// Generate Gaussian Convolution Kernel
// Created by Haomin Kong on 23-12-13.
// https://github.com/a645162/HomeWorkNewYearVideo

#include "GaussianKernel.h"

#include <cmath>

#include "../../Config/ConstVar.h"

float* createGaussianKernel(int size, float strength) {
    auto* kernel = static_cast<float *>(malloc(size * size * sizeof(float)));

    float sum = 0;

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            const auto value =
                    static_cast<float>(
                        static_cast<double>(1 / (2 * M_PI * std::pow(strength, 2))) *
                        std::exp(
                            -(
                                std::pow(static_cast<float>(x) - static_cast<float>(size - 1) / 2.0f, 2) +
                                std::pow(static_cast<float>(y) - static_cast<float>(size - 1) / 2.0f, 2)
                            )
                            /
                            (2 * std::pow(strength, 2))
                        )
                    );
            kernel[x * size + y] = value;

            sum += value;
        }
    }

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            kernel[x * size + y] /= sum;
        }
    }

    return kernel;
}

float* createGaussianKernel(GaussianKernelParameters parameters) {
    return createGaussianKernel(parameters.size, parameters.strength);
}

GaussianKernelParameters getGaussianKernelParameters(int size, float strength) {
    return {
        size,
        strength
    };
}

GaussianKernelParameters calcGaussianKernelParameters(float blur_intensity) {
    if (blur_intensity > 100) {
        blur_intensity = 100;
    }
    if (blur_intensity < 0) {
        blur_intensity = 0;
    }

    // Interpolate between [0, 100] to get the maximum blur size [1, 31]
    const int max_blur_size =
            static_cast<int>(std::round((blur_intensity - 0) * (31 - 1) / (100 - 0) + 1));

    // Interpolate between [0, 100] to get the maximum blur strength [0.1, 10.0]
    const auto max_blur_strength =
            static_cast<float>((blur_intensity - 0) * (10.0 - 0.1) / (100 - 0) + 0.1);

    return getGaussianKernelParameters(max_blur_size, max_blur_strength);
}
