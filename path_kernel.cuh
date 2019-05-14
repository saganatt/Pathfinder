#ifndef PATH_KERNEL_H
#define PATH_KERNEL_H

#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void pathKernel(uint32_t *d_array, bool *d_path_array, uint32_t arraySize,
        uint32_t currentDiagSize, bool isAboveMainDiag) {
    const uint32_t blockId = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    const uint32_t tid = blockId * blockDim.x + threadIdx.x;

    if(tid >= currentDiagSize) {
    	return;
    }

    uint32_t column = tid;
    uint32_t row = currentDiagSize - column - 1;

    if(!isAboveMainDiag) {
        column = arraySize - currentDiagSize + tid;
        row = arraySize - 1 - tid;
    }

    if (row > 0 && ((column > 0 && d_array[(row - 1) * arraySize + column] < d_array[row * arraySize + column - 1])
                    || column == 0)) {
        d_array[row * arraySize + column] += d_array[(row - 1) * arraySize + column]; // From the top
        d_path_array[row * arraySize + column] = true;
    } else if (column > 0) {
        d_array[row * arraySize + column] += d_array[row * arraySize + column - 1]; // From the left
        d_path_array[row * arraySize + column] = false;
    }
}

#endif