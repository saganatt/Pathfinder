#ifndef ARRAY_GEN_KERNEL_H
#define ARRAY_GEN_KERNEL_H

#include <cuda_runtime.h>

__global__ void arrayGenKernel(uint32_t *d_array, uint32_t arraySize) {
    const uint32_t blockId = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    const uint32_t tid = blockId * blockDim.x + threadIdx.x;

    if(tid < arraySize * arraySize) {
        d_array[tid] = tid;
    }
}

#endif