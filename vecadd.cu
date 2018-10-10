#include<cstdio>
#include<cuda.h>

__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

const int MAXN = 1024*1024;

int main() {
    size_t size = MAXN * sizeof(float);
    float *a = (float*)malloc(size);
    float *b = (float*)malloc(size);
    float *c = (float*)malloc(size);
    for (int i = 0; i < MAXN; ++i) a[i] = i;
    for (int i = 0; i < MAXN; ++i) b[i] = i;
    printf("size: %ld\n", size);

    float* dA;
    cudaMalloc((void **)&dA, size);
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) {
        printf("ERROR: %d %s\n", e2, cudaGetErrorString(e2));
    }

    cudaMemcpy(dA, a, size, cudaMemcpyHostToDevice);
    float* dB;
    cudaMalloc((void **)&dB, size);
    cudaMemcpy(dB, b, size, cudaMemcpyHostToDevice);
    float* dC;
    cudaMalloc((void **)&dC, size);

    clock_t start_d = clock();
    cudaDeviceSynchronize();
    VecAdd<<<(MAXN+255)/256, 256>>>(dA, dB, dC, MAXN);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %d %s\n", error, cudaGetErrorString(error));
    }
    clock_t end_d = clock();
    printf("clock %lf\n", (double)(end_d-start_d)/CLOCKS_PER_SEC);

    cudaMemcpy(c, dC, size, cudaMemcpyDeviceToHost);
    printf("%f %f %f\n", a[1024], b[1024], c[1024]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
