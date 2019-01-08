// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// includes CUDA
//#include <cuda_runtime.h>
#include <helper_string.h>
#include <helper_timer.h>

#include <cpu_version.h>

const uint32_t defaultArraySize = 4;
const uint32_t maxArrayPrintSize = 16;

void parseArguments(int argc, char **argv, uint32_t *arraySize, bool *useGPU);
void printArguments(uint32_t arraySize, bool useGPU);
void usage(const char* name);

uint32_t* generateArray(uint32_t arraySize);
void printArray(uint32_t* arr, uint32_t arraySize);

int main(int argc, char **argv) {
    printf("\n%s starting...\n\n", argv[0]);

    uint32_t arraySize = defaultArraySize;
    bool useGPU = true;
    parseArguments(argc, argv, &arraySize, &useGPU);
    printArguments(arraySize, useGPU);

    uint32_t* h_array = generateArray(arraySize);
    if(arraySize <= maxArrayPrintSize) {
        printArray(h_array, arraySize);
    }

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    uint32_t result = UINT_MAX;
    if(useGPU) {
        runGPU(arraySize, h_array);
    }
    else {
        result = runCPU(arraySize, h_array);
    }

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    printf("Calculated shortest path: %u\n", result);

    free(h_array);

    exit(EXIT_SUCCESS);
}

void parseArguments(int argc, char **argv, uint32_t *arraySize, bool *useGPU) {
    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **) argv, "s")) {
            *arraySize = getCmdLineArgumentInt(argc, (const char **) argv, "s");
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "array-size")) {
            *arraySize = getCmdLineArgumentInt(argc, (const char **) argv, "array-size");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "use-gpu")) {
            *useGPU = true;
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "use-cpu")) {
            *useGPU = false;
        }
        else {
            printf("Use of either --use-gpu or --use-cpu is required\n");
            usage(argv[0]);
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "help")) {
            usage(argv[0]);
        }
    }
}

void usage(const char* name) {
    printf("Usage: %s [OPTIONS=VALUE]\n", name);
    printf("Available options:\n");
    printf("\t-s, --array-size\t\tset array size\tdefault: %u\n", defaultArraySize);
    printf("\t    --use-gpu\t\t\trun GPU version of the program\n");
    printf("\t    --use-cpu\t\t\trun CPU version of the program\n");
    printf("\t    --help\t\t\tdisplay this help and exit\n");
    exit(EXIT_FAILURE);
}

void printArguments(uint32_t arraySize, bool useGPU) {
    printf("Arguments set:\n");
    printf("Array size:\t\t%u x %u\n", arraySize, arraySize);
    printf("Program version:\t%s\n", useGPU ? "GPU" : "CPU");
    printf("\n");
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
     printf("Printing array\n");
     for (uint32_t i = 0; i < arraySize; i++) {
         for (uint32_t j = 0; j < arraySize; j++) {
             printf("%5u ", arr[i * arraySize + j]);
         }
         printf("\n");
     }
     printf("\n");
 }
