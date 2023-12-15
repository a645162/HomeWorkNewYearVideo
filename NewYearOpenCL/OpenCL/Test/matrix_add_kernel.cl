// matrix_add_kernel.cl
// https://blog.csdn.net/pengx17/article/details/7876657
// OpenCL kernel to compute the sum of matrix elements

inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void matrixSum(__global float* matrix, __global float* result, const int rows, const int cols) {
    int global_id = get_global_id(0);
    int row = global_id / cols;
    int col = global_id % cols;

    // Check if the work-item is within the matrix bounds
    if (row < rows && col < cols) {
        AtomicAdd(&result[0], matrix[row * cols + col]);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
