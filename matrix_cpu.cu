#include <cstdio>
#include <cstring>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16

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
    memset(c.elements, 0, sizeof(MATRIX_SIZE*MATRIX_SIZE*sizeof(float)));
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            for (int t = 0; t < MATRIX_SIZE; ++t)
                C.elements[i*MATRIX_SIZE+j] += A.elements[i*MATRIX_SIZE+t] * B.elements[i+t*MATRIX_SIZE];
    printf("%f\n", C.elements[3*A.width+3]);
}

int main() {
    Init();
    MatMul(a, b, c);
    return 0;
}
