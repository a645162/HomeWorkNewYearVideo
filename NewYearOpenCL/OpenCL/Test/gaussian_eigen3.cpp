//#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <cmath>
#include <cstdio>

#ifdef __APPLE__
#include "Eigen/Dense"
#else
#include <eigen3/Eigen/Dense>
#endif

#include <chrono>

#include <cmath>
#include <immintrin.h>

#define M_PI 3.14159265358979323846

#define SIZE 28

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

    return kernel / kernel.sum();
}

int main() {

    int size = SIZE; // Adjust the size as needed
    float strength = 9.1f; // Adjust the strength as needed

    auto time1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXf gaussianKernel = createGaussianKernel(size, strength);
    auto time2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
    std::cout << "(Eigen)Time taken to create the Gaussian kernel: " << duration1 << " microseconds" << std::endl;

    auto time3 = std::chrono::high_resolution_clock::now();
    float *kernel;
    kernel = (float *) malloc(size * size * sizeof(float));
    float sum = 0;
//#pragma omp parallel for num_threads(1)
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            float value = (1 / (2 * M_PI * std::pow(strength, 2))) *
                          std::exp(-((x - (size - 1) / 2.0f) * (x - (size - 1) / 2.0f) +
                                     (y - (size - 1) / 2.0f) * (y - (size - 1) / 2.0f)) /
                                   (2 * std::pow(strength, 2)));
            kernel[x * size + y] = value;
            sum += value;
        }
    }
//#pragma omp parallel for num_threads(1)
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            kernel[x * size + y] /= sum;
        }
    }

    auto time4 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count();
    std::cout << "(For)Time taken to create the Gaussian kernel: " << duration2 << " microseconds" << std::endl;


    std::cout << "Sum: " << sum << std::endl;

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
            data[i * size + j] -= kernel[i * size + j];
        }
    }

    std::cout << "diff: " << std::endl;
    float diff = gaussianKernel.sum();
    double roundedNumber = round(diff * 100000) / 100000;

    std::cout << roundedNumber << std::endl;

//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            printf("%.6f  ", data[i * size + j]);
//        }
//        printf("\n");
//    }

    free(kernel);
    return 0;
}
