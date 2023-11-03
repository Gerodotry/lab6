#include <iostream>
#include <mpi.h>

#define N 4

void matrix_multiply(int A[N][N], int B[N][N], int C[N][N], int start_row, int end_row)
{
    for (int i = start_row; i < end_row; i++)
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

void print_matrix(int C[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
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

    // Gather all parts of matrix C to rank 0
    int sendcounts[size];
    int displacements[size];
    int recvcounts[size];

    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = rows_per_thread * N;
        recvcounts[i] = rows_per_thread * N;
        displacements[i] = i * rows_per_thread * N;
    }

    int C_recv[N][N] = {0}; // Separate buffer for rank 0

    MPI_Gatherv(&C[start_row][0], sendcounts[rank], MPI_INT, &C_recv[0][0], recvcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    double max_elapsed_time;

    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Optimized Fox Algorithm Results:\n";
        std::cout << "Processes\tSizes\tElapsed Time\n";
        for (int num_processes = 1; num_processes <= size; num_processes++)
        {
            std::cout << num_processes << "\t";
            std::cout << N << "\t";
            std::cout << max_elapsed_time << " seconds\n";
        }
        std::cout << "Matrix C:\n";
        print_matrix(C_recv);
    }

    MPI_Finalize();

    return 0;
}
