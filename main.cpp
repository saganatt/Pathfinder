#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include <helper_string.h>
#include <cuda_runtime.h>

#include <cpu_version.h>

const uint32_t defaultArraySize = 16384; // 2 ^ 14
const uint32_t maxArraySize = 32768; // 2 ^ 15 due to size_t limitations

extern "C" void runGPU(uint32_t arraySize, uint32_t *result, bool *result_path);

void parseArguments(int argc, char **argv, uint32_t *arraySize, bool *useGPU, bool *verbose);
void printArguments(uint32_t arraySize, bool useGPU, bool verbose);
void usage(const char* name);

int main(int argc, char **argv) {
    printf("\nPathfinder starting...\n\n");

    uint32_t arraySize = defaultArraySize;
    bool useGPU = true, verbose = false;
    parseArguments(argc, argv, &arraySize, &useGPU, &verbose);
    printArguments(arraySize, useGPU, verbose);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    uint32_t result = UINT_MAX;
    bool *result_path = new bool[arraySize * 2 - 2];

    if(useGPU) {
        runGPU(arraySize, &result, result_path);
    }
    else {
        runCPU(arraySize, verbose, result, result_path);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nTotal processing time:  %3.3f ms \n", time);

    printf("Calculated shortest path: %u\n", result);
    if(verbose) {
        printf("Path trace:\n");
        for(uint32_t i = 0; i < arraySize * 2 - 2; i++) {
            printf("|%c", result_path[i] ? '^' : '<');
        }
        printf("\n");
    }

    free(result_path);

    exit(EXIT_SUCCESS);
}

void parseArguments(int argc, char **argv, uint32_t *arraySize, bool *useGPU, bool *verbose) {
    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **) argv, "s")) {
            *arraySize = getCmdLineArgumentInt(argc, (const char **) argv, "s");
            if(*arraySize > maxArraySize) {
                printf("Array size cannot be bigger than: %u\n\n", maxArraySize);
                usage(argv[0]);
            }
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "array-size")) {
            *arraySize = getCmdLineArgumentInt(argc, (const char **) argv, "array-size");
            if(*arraySize > maxArraySize) {
                printf("Array size cannot be bigger than: %u\n\n", maxArraySize);
                usage(argv[0]);
            }
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "v") ||
            checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
            *verbose = true;
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "use-gpu")) {
            *useGPU = true;
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "use-cpu")) {
            *useGPU = false;
        }
        else {
            printf("Use of either --use-gpu or --use-cpu is required\n\n");
            usage(argv[0]);
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "help")) {
            usage(argv[0]);
        }
    }
}

void usage(const char* name) {
    printf("Usage: %s [OPTION [=VALUE]]\n", name);
    printf("Available options:\n");
    printf("s, --array-size\t\tset array size, default:\t%u\n", defaultArraySize);
    printf("\t\t\t\tmax possible value:\t\t%u\n", maxArraySize);
    printf("    --use-gpu\t\t\trun GPU version of the program\n");
    printf("    --use-cpu\t\t\trun CPU version of the program\n");
    printf("v, --verbose\t\t\tprint result path\n");
    printf("    --help\t\t\tdisplay this help and exit\n");
    exit(EXIT_FAILURE);
}

void printArguments(uint32_t arraySize, bool useGPU, bool verbose) {
    printf("Arguments set:\n");
    printf("Array size:\t\t%u x %u\n", arraySize, arraySize);
    printf("Program version:\t%s\n", useGPU ? "GPU" : "CPU");
    printf("Verbose:\t\t%s\n", verbose ? "true" : "false");
    printf("\n");
}