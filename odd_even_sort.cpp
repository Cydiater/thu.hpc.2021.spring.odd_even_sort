#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "worker.h"

void merge(float *a, int na, float *b, int nb, float *merged, int offset) {
	int cl = offset, cr = offset + na - 1;
	if (cl == 0) {
		int pp = 0, qq = 0;
		for (int i = cl; i <= cr; i++) {
			if (qq == nb) merged[i - cl] = a[pp++];
			else if (pp == na) merged[i - cl] = b[qq++];
			else merged[i - cl] = a[pp] < b[qq] ? a[pp++] : b[qq++];
		}
	} else {
		int pp = na - 1, qq = nb - 1;
		for (int i = cr; i >= cl; i--) {
			if (qq == -1) merged[i - cl] = a[pp--];
			else if (pp == -1) merged[i - cl] = b[qq--];
			else merged[i - cl] = a[pp] > b[qq] ? a[pp--] : b[qq--];
		}
	}
}

void Worker::sort() {
	if (out_of_range)
		return;
	// variables that to be used
	int round = -1;
	size_t block_size = ceiling(n, nprocs);
	float *dataFromNeighbor = new float[block_size];
	float *merged = new float[block_size];
	MPI_Status status[2];
	MPI_Request request[2];

	// do std::sort initially to make local data in order
	std::sort(data, data + block_len);

	// odd even sort phrases
	while (++round < nprocs) {
		// setup the index of the neighbor
		bool isMaster = (round % 2) == (rank % 2);
		int neighborIndex = isMaster ? rank + 1 : rank - 1, myOffset = 0, neighborLen;
		if (neighborIndex < 0 || (neighborIndex > rank && last_rank) )
			continue;

		MPI_Isend(data, block_len, MPI_FLOAT, neighborIndex, round, MPI_COMM_WORLD, request + 0);
		MPI_Irecv(dataFromNeighbor, block_size, MPI_FLOAT, neighborIndex, round, MPI_COMM_WORLD, request + 1);
		MPI_Wait(request + 1, status + 1);
		MPI_Get_count(status + 1, MPI_FLOAT, &neighborLen);
		if (neighborIndex < rank) myOffset += neighborLen;
		merge(data, block_len, dataFromNeighbor, neighborLen, merged, myOffset);
		std::swap(data, merged);
		MPI_Wait(request + 0, MPI_STATUS_IGNORE);
	}

	// release memory
	delete[] dataFromNeighbor;
	delete[] merged;
}
