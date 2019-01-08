#ifndef CPU_VERSION_H
#define CPU_VERSION_H

void printResultOnArray(uint32_t arraySize, uint32_t* array, std::vector<bool> path);

uint32_t runCPU(uint32_t arraySize, uint32_t* h_array) {
    uint32_t **tmp = new uint32_t *[arraySize];
    bool **path = new bool *[arraySize]; // true means coming from the top, false - from the left

    printf("Starting CPU calculations...\n");

    for (uint32_t i = 0; i < arraySize; i++) {
        tmp[i] = new uint32_t[arraySize];
        path[i] = new bool[arraySize];
        for (uint32_t j = 0; j < arraySize; j++) {
            tmp[i][j] = h_array[i * arraySize + j];
            path[i][j] = false;

            if (i > 0 && ((j > 0 && tmp[i - 1][j] < tmp[i][j - 1]) || j == 0)) {
                tmp[i][j] += tmp[i - 1][j]; // From the top
                path[i][j] = true;
            } else if (j > 0) {
                tmp[i][j] += tmp[i][j - 1]; // From the left
                path[i][j] = false;
            }
        }
    }

    std::vector<bool> pathNodes;
    uint32_t i = arraySize, j = arraySize;
    while(i > 0 && j > 0) {
        pathNodes.push_back(path[i - 1][j - 1]);
        if(path[i - 1][j - 1]) i--;
        else j--;
    }

    printResultOnArray(arraySize, h_array, pathNodes);

    return tmp[arraySize - 1][arraySize - 1];
}

void printResultOnArray(uint32_t arraySize, uint32_t* array, std::vector<bool> path) {
    uint32_t pathX = 0, pathY = 0;
    uint32_t k = path.size() - 1;
    printf("Computed path:\n");
    for(uint32_t i = 0; i < arraySize; i++) {
        for(uint32_t j = 0; j < arraySize; j++) {
            if(i == pathX && j == pathY) {
                printf("|%5u", array[i * arraySize + j]);
            }
            else {
                printf("|#####");
            }
            if(k > 0 && !path[k - 1]) {
                pathY++;
                k--;
            }
        }
        if(k > 0 && path[k - 1]) {
            pathX++;
            k--;
        }
        printf("|\n");
    }
    printf("\n");
}

#endif