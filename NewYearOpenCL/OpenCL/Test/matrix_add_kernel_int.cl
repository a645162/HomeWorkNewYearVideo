// matrix_add_kernel_int.cl

// OpenCL kernel to compute the sum of matrix elements for integers
__kernel void matrixSum(__global int* matrix, __global int* result, const int rows, const int cols) {
    int global_id = get_global_id(0);
    int row = global_id / cols;
    int col = global_id % cols;

    // Check if the work-item is within the matrix bounds
    if (row < rows && col < cols) {
        atomic_add(&result[0], matrix[row * cols + col]);
    }
}
