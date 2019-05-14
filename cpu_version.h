#ifndef CPU_VERSION_H
#define CPU_VERSION_H

#include <stdint.h>

const int maxPrintArraySize = 20;

uint32_t* generateArray(uint32_t arraySize);
void printArray(uint32_t* arr, uint32_t arraySize);
void generatePathFromPathArray(bool *pathArray, uint32_t arraySize, bool *resultPath);

void runCPU(uint32_t arraySize, bool verbose, uint32_t& result, bool* resultPath) {
    uint32_t *array = generateArray(arraySize);

    if(verbose && arraySize < maxPrintArraySize) {
        printArray(array, arraySize);
    }

    bool *pathArray = new bool [arraySize * arraySize];

    for (uint32_t i = 0; i < arraySize; i++) {
        for (uint32_t j = 0; j < arraySize; j++) {
            pathArray[i * arraySize + j] = false;

            if (i > 0 && ((j > 0 && array[(i - 1) * arraySize + j] < array[i * arraySize + j - 1]) || j == 0)) {
                array[i * arraySize + j] += array[(i - 1) * arraySize + j]; // From the top
                pathArray[i * arraySize + j] = true;
            } else if (j > 0) {
                array[i * arraySize + j] += array[i * arraySize + j - 1]; // From the left
                pathArray[i * arraySize + j] = false;
            }
        }
    }
    
    generatePathFromPathArray(pathArray, arraySize, resultPath);
    result = array[(arraySize - 1) * arraySize + arraySize - 1];

    free(array);
    free(pathArray);
}

uint32_t* generateArray(uint32_t arraySize) {
    uint32_t *arr = new uint32_t[arraySize * arraySize];
    for (uint32_t i = 0; i < arraySize; i++) {
        for (uint32_t j = 0; j < arraySize; j++) {
            arr[i * arraySize + j] = i * arraySize + j;
        }
    }
    return arr;
}

void printArray(uint32_t* arr, uint32_t arraySize) {
    for (uint32_t i = 0; i < arraySize; i++) {
        for (uint32_t j = 0; j < arraySize; j++) {
            printf("|%5u ", arr[i * arraySize + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void generatePathFromPathArray(bool *pathArray, uint32_t arraySize, bool *resultPath) {
    uint32_t i = arraySize, j = arraySize;
    uint32_t k = arraySize * 2 - 2 - 1;
    while(k > 0) {
        resultPath[k] = pathArray[(i - 1) * arraySize + j - 1];
        k--;
        if(pathArray[(i - 1) * arraySize + j - 1]) i--;
        else j--;
    }
}

#endif