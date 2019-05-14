#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <path_kernel.cuh>
#include <array_gen_kernel.cuh>

const int warpSize = 32;
const int warpsPerBlock = 32;
//const int maxWarpsPerBlock = 32;
//const int maxThreadsPerBlock = 1024;
const int maxGridDimxy = 1024;
const int maxGridDimz = 64;

extern "C" void runGPU(uint32_t arraySize, uint32_t *result, bool *resultPath);

__host__ void calculateBlockAndGridDims(uint32_t numberOfThreads, dim3 *blocks, dim3 *threads);
__host__ void generateArrayParallel(uint32_t arraySize, uint32_t *d_array);
__host__ void generatePathFromPathArrayKernel(bool *pathArray, uint32_t arraySize, bool *resultPath);

extern "C" void runGPU(uint32_t arraySize, uint32_t *result, bool *resultPath) {
    // -1 for not processing first and last field
    uint32_t halfOfSteps = arraySize - 1;
    size_t arrayMemSize = arraySize * arraySize * sizeof(uint32_t);
    size_t pathArrayMemSize = arraySize * arraySize * sizeof(bool);

    uint32_t currentDiagSize = 1;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    uint32_t *d_array;
    checkCudaErrors(cudaMalloc((void **) &d_array, arrayMemSize));
    generateArrayParallel(arraySize, d_array);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Array generation time:  %3.3f ms \n", time);

    cudaEventRecord(start, 0);

    bool *h_path_array = new bool[arraySize * arraySize];
    bool *d_path_array;
    checkCudaErrors(cudaMalloc((void **) &d_path_array, pathArrayMemSize));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Memory alloc & copying elapsed time:  %3.3f ms \n", time);

    cudaEventRecord(start, 0);

    dim3 blocks, threads;

   for (uint32_t i = 0; i < halfOfSteps; i++) {
       currentDiagSize++;
        calculateBlockAndGridDims(currentDiagSize, &blocks, &threads);
        pathKernel << < blocks, threads >> > (d_array, d_path_array, arraySize, currentDiagSize, true);
    }

    for(uint32_t i = 0; i < halfOfSteps; i++) {
        currentDiagSize--;
        calculateBlockAndGridDims(currentDiagSize, &blocks, &threads);
        pathKernel << < blocks, threads >> > (d_array, d_path_array, arraySize, currentDiagSize, false);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("All kernel calls elapsed time:  %3.3f ms \n", time);

    cudaEventRecord(start, 0);

    checkCudaErrors(cudaMemcpy(result, d_array + arraySize * arraySize - 1, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_path_array, d_path_array, pathArrayMemSize,
                               cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Memory copying back elapsed time:  %3.3f ms \n", time);

    generatePathFromPathArrayKernel(h_path_array, arraySize, resultPath);

    free(h_path_array);
    checkCudaErrors(cudaFree(d_array));
    checkCudaErrors(cudaFree(d_path_array));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__host__ void calculateBlockAndGridDims(uint32_t numberOfThreads, dim3 *blocks, dim3 *threads) {
    uint32_t blockSize = numberOfThreads < warpSize * warpsPerBlock ?
                         numberOfThreads : warpSize * warpsPerBlock;
    *threads = dim3(blockSize, 1, 1);

    uint32_t numberOfBlocks = (numberOfThreads + blockSize - 1) / blockSize;
    uint32_t gridDimx = numberOfBlocks < maxGridDimxy ? numberOfBlocks : maxGridDimxy;

    numberOfBlocks = (numberOfBlocks + gridDimx - 1) / gridDimx;
    uint32_t gridDimy = numberOfBlocks < maxGridDimxy ? numberOfBlocks : maxGridDimxy;

    numberOfBlocks = (numberOfBlocks + gridDimy - 1) / gridDimy;
    if(numberOfBlocks > maxGridDimz) {
        printf("Requested array and accuracy demands unachievable amount of working blocks and threads.\n");
        printf("Consider narrowing your search to smaller area, or a smaller array\n\n");
        printf("Exiting program with failure code\n");
        exit(EXIT_FAILURE);
    }
    uint32_t gridDimz = numberOfBlocks;

    *blocks = dim3(gridDimx, gridDimy, gridDimz);
}

__host__ void generateArrayParallel(uint32_t arraySize, uint32_t *d_array) {
    dim3 blocks, threads;
    calculateBlockAndGridDims(arraySize * arraySize, &blocks, &threads);

    arrayGenKernel << < blocks, threads >> > (d_array, arraySize);
}

__host__ void generatePathFromPathArrayKernel(bool *pathArray, uint32_t arraySize, bool *resultPath) {
    uint32_t i = arraySize, j = arraySize;
    uint32_t k = arraySize * 2 - 2 - 1;
    while(k > 0) {
        resultPath[k] = pathArray[(i - 1) * arraySize + j - 1];
        k--;
        if(pathArray[(i - 1) * arraySize + j - 1]) i--;
        else j--;
    }
}