#include <iostream>
#include <curand_kernel.h>

// CUDA kernel to generate random numbers
__global__ void generateRandomNumbers(float* data, int size, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize the random number generator state
    curandState state;
    curand_init(seed, tid, 0, &state);


    //Generate uniformly distributed random numbers in the range [0, 1)
    data[tid] = curand_uniform(&state);

}

int main() {
    const int size = 1000; // Size of your array
    float* device_data;
    cudaMalloc((void**)&device_data, size * sizeof(float));

    // Set the seed for the random number generator
    unsigned long long seed = 123;

    // Call the CUDA kernel to generate random numbers
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    generateRandomNumbers<<<gridSize, blockSize>>>(device_data, size, seed);

    // Copy the results back to the host
    float* host_data = new float[size];
    cudaMemcpy(host_data, device_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first 10 random numbers
    std::cout << "First 10 random numbers: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << host_data[i] << " ";
    }

    // Free memory
    delete[] host_data;
    cudaFree(device_data);

    return 0;
}
