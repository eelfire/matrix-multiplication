// matrix multiplication in parallel
    // simple
    // using fork
    // using pthreads
    // using openmp

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048

float A[N][N], B[N][N], C[N][N];

void init_matrix(float m[N][N]) {
    int i, j;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            m[i][j] = (float)rand()/(float)RAND_MAX;
        }
    }
}

void print_head_tail(float m[N][N]) {
    // print first and last row of matrix
    int i, j;
    for (i=0; i<N; i = i + N-1) {
        for (j=0; j<N; j++) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiplication(float a[N][N], float b[N][N]) {
    // multiply matrix a and b and store in c
    int i, j, k;
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++) {
                C[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    srand(time(NULL));
    init_matrix(A);
    init_matrix(B);
    // init_matrix(C);

    clock_t start, end;
    // double cpu_time_used;
    start = clock();
    matrix_multiplication(A, B);
    end = clock();

    // print_head_tail(C);
    printf("Time taken: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    
    return 0;
}