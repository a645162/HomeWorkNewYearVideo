//#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <chrono>

Eigen::MatrixXf createGaussianKernel(int size, float strength) {
    Eigen::MatrixXf kernel(size, size);

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            float value = (1 / (2 * M_PI * std::pow(strength, 2))) *
                          std::exp(-((x - (size - 1) / 2.0f) * (x - (size - 1) / 2.0f) +
                                     (y - (size - 1) / 2.0f) * (y - (size - 1) / 2.0f)) /
                                   (2 * std::pow(strength, 2)));

            kernel(x, y) = value;
        }
    }

//    float sum=0;
//    for (int x = 0; x < size; ++x) {
//        for (int y = 0; y < size; ++y) {
//            sum+=kernel(x, y);
//        }
//    }
//
//    for (int x = 0; x < size; ++x) {
//        for (int y = 0; y < size; ++y) {
//            kernel(x, y) /= sum;
//        }
//    }
//    return kernel;
    return kernel / kernel.sum();
}

int main() {
    int size = 28; // Adjust the size as needed
    float strength = 9.1f; // Adjust the strength as needed

    auto time1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXf gaussianKernel = createGaussianKernel(size, strength);
    auto time2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "Time taken to create the Gaussian kernel: " << duration1 << " microseconds" << std::endl;


    // Print the resulting Gaussian kernel
//    std::cout << "Gaussian Kernel:" << std::endl << gaussianKernel << std::endl;

//    for (int x = 0; x < size; ++x) {
//        for (int y = 0; y < size; ++y) {
////            std::cout << gaussianKernel(x, y) << " ";
//            printf("%.6f  ", gaussianKernel(x, y));
//        }
//        std::cout << std::endl;
//    }

    float *data = gaussianKernel.data();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.6f  ", data[i * size + j]);
        }
        printf("\n");
    }

    return 0;
}
