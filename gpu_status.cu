#include <stdio.h> 

struct cudaFuncAttributes funcAttrib;

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaFuncGetAttributes(&funcAttrib, KERNEL);
    printf("%s numRegs=%d\n",KERNELNAME,funcAttrib.numRegs);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        int support = 0;
        cudaDeviceGetAttribute(&support, cudaDevAttrStreamPrioritiesSupported, 0);
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Priorities support: %d\n", support);
        if (support) {
          int leastP = 0, greatP = 0;
          cudaDeviceGetStreamPriorityRange(&leastP, &greatP);
          printf("leastPriority %d, greatestPriority %d\n", leastP, greatP);
        }
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        // printf("%d\n", prop.maxThreadsDim);
        printf("%d\n", prop.maxThreadsPerBlock);
    }
    return 0;
}
