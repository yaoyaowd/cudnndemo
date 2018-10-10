#include <cstdio>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
    float cv = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < A.width; ++i)
        cv += A.elements[row * A.width + i] * B.elements[i * B.width + col];
    C.elements[row * C.width + col] = cv;
}

Matrix a, b, c;

void Init() {
    a.width = a.height = MATRIX_SIZE;
    b.width = b.height = MATRIX_SIZE;
    a.elements = (float*) malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
    b.elements = (float*) malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            a.elements[i*MATRIX_SIZE+j] = b.elements[i*MATRIX_SIZE+j] = i + j;
    c.width = c.height = MATRIX_SIZE;
    c.elements = (float*) malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix dA;
    dA.width = A.width;
    dA.height = A.height;
    size_t size = A.width * B.height * sizeof(float);
    cudaMalloc(&dA.elements, size);
    cudaMemcpy(dA.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix dB;
    dB.width = B.width;
    dB.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&dB.elements, size);
    cudaMemcpy(dB.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix dC;
    dC.width = C.width;
    dC.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&dC.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);
    MatMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);

    cudaDeviceSynchronize();
    cudaMemcpy(C.elements, dC.elements, size, cudaMemcpyDeviceToHost);

    printf("%f\n", C.elements[3*A.width+3]);

    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC.elements);
}

int main() {
    Init();
    MatMul(a, b, c);
    return 0;
}
