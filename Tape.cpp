#include <iostream>
#include <mpi.h>

#define N 4

void matrix_multiply(int A[N][N], int B[N][N], int C[N][N], int start, int end)
{
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[N][N] = {{1, 2, 3, 4},
                   {5, 6, 7, 8},
                   {9, 10, 11, 12},
                   {13, 14, 15, 16}};

    int B[N][N] = {{17, 18, 19, 20},
                   {21, 22, 23, 24},
                   {25, 26, 27, 28},
                   {29, 30, 31, 32}};

    int C[N][N] = {{0}};

    double start_time, end_time;
    double elapsed_time;

    int rows_per_thread = N / size;
    int start_row = rank * rows_per_thread;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_thread;

    start_time = MPI_Wtime();

    matrix_multiply(A, B, C, start_row, end_row);

    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    double max_elapsed_time;

    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Tape Algorithm Results:\n";
        std::cout << "Processes\tSizes\tElapsed Time\n";
        for (int num_processes = 1; num_processes <= size; num_processes++)
        {
            std::cout << num_processes << "\t";
            std::cout << N << "\t";
            std::cout << max_elapsed_time << " seconds\n";
        }
    }

    MPI_Finalize();

    return 0;
}
