// matrix multiplication in parallel
// simple
// using pthread
// using fork
// using pthreads
// using openmp

#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define N 2048
#define NUM_THREADS 10

sem_t semaphores[NUM_THREADS];

float A[N][N], B[N][N], C[N][N];
float (*fC)[N];

void init_matrix_random(float m[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            m[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void init_matrix_row_col(float m[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            m[i][j] = i + j;
        }
    }
}

void init_matrix_with_zero(float m[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            m[i][j] = 0;
        }
    }
}

void print_head_tail(float m[N][N]) {
    // print first and last row of matrix
    int i, j;
    for (i = 0; i < N; i = i + N - 1) {
        for (j = 0; j < N; j++) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiplication() {
    // multiply matrix a and b and store in c
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

typedef struct {
    int row;
    int col;
} thread_data;

void *matrix_multiplication_pthread_logic(void *arg) {
    // multiply matrix a and b and store in c using pthread and semaphores
    int k;
    thread_data *data = (thread_data *)arg;
    int row = data->row;
    int col = data->col;

    // printf("%d %d\n", row, col);

    if (col >= N) {
        row++;
        col = 0;
    }

    // pthread exit
    if (row >= N) {
        pthread_exit(NULL);
    }

    thread_data next_data = {row, col + 1};

    if (C[row][col] == 0) {
        sem_wait(&semaphores[col % NUM_THREADS]);
        printf("%d %d\n", row, col);
        for (k = 0; k < N; k++) {
            C[row][col] += A[row][k] * B[k][col];
            // printf("%f ", C[row][col]);
        }
        sem_post(&semaphores[col % NUM_THREADS]);
    }

    matrix_multiplication_pthread_logic(&next_data);

    // return NULL;
    pthread_exit(NULL);
}

void matrix_multiplication_pthread() {
    // helper function to create semaphores, threads and call
    // matrix_multiplication_pthread_logic
    int i;
    pthread_t threads[NUM_THREADS];
    // sem_t semaphores[NUM_THREADS];

    thread_data data = {0, 0};

    for (i = 0; i < NUM_THREADS; i++) {
        sem_init(&semaphores[i], 0, 1);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, matrix_multiplication_pthread_logic,
                       &data);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    for (i = 0; i < NUM_THREADS; i++) {
        sem_destroy(&semaphores[i]);
    }
}

void multiply_row_col(int row, int col) {
    for (int k = 0; k < N; k++) {
        fC[row][col] += A[row][k] * B[k][col];
    }
}

void matrix_multiplication_fork() {
    // multiply matrix A and B and store in fC using fork
    fC = mmap(NULL, N * N * sizeof(float), PROT_READ | PROT_WRITE,
              MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    pid_t child_pid;

    for (int i = 0; i < N; i++) {
        child_pid = fork();

        if (child_pid == -1) {
            perror("fork");
            exit(EXIT_FAILURE);
        }

        if (child_pid == 0) { // Child process
            for (int j = 0; j < N; j++) {
                multiply_row_col(i, j);
            }
            exit(EXIT_SUCCESS);
        }
    }

    // Wait for all child processes to finish
    for (int i = 0; i < N; i++) {
        wait(NULL);
    }
}

int main() {
    clock_t main_start, main_end;
    main_start = clock();

    struct timeval m_start, m_end;
    gettimeofday(&m_start, 0);

    srand(time(NULL));
    // init_matrix_random(A);
    // init_matrix_random(B);
    init_matrix_row_col(A);
    init_matrix_row_col(B);
    // init_matrix(C);
    init_matrix_with_zero(C);

    clock_t start, end;
    // double cpu_time_used;
    start = clock();

    // simple
    // matrix_multiplication();

    // using pthread
    // matrix_multiplication_pthread(); // isn't work for large matrix

    // using fork
    matrix_multiplication_fork();

    end = clock();

    print_head_tail(fC);
    printf("Time taken: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    munmap(fC, N * N * sizeof(float));

    gettimeofday(&m_end, 0);
    double total = m_end.tv_sec - m_start.tv_sec;
    if (m_end.tv_usec - m_start.tv_usec < 0) total++;
    total *= 1000*1000;
    total += m_end.tv_usec - m_start.tv_usec;
    printf("Execution time: %f\n", total);

    main_end = clock();
    printf("Time taken (main): %f\n", ((double)(main_end - main_start)) / CLOCKS_PER_SEC);
    printf("%ld\n", main_end);

    printf("%ld\n", sizeof(uint));
    return 0;
}
