#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cout << "Cores number = " << world_rank << "/" << world_size <<endl;
    MPI_Finalize();
    
    return 0;
}


