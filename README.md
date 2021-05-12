[TOC]

# 实验一：奇偶排序

## 源代码与简要说明

### 源代码

```cpp
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
```

### 简要说明

我的实现主要进行了以下的操作：

1. 为接受相邻进程的数据以及后续的归并分配空间，并初始化将要用到的一些变量例如 `MPI_Status` 等。
2. 在本地进行初始化的排序，这一步其实可以进行一些算法上的优化例如采用基数排序等，但是并没有这个必要。
3. 进行 $n$ 轮迭代，其中 $n$ 为分配的线程的数量，不难发现这是奇偶排序的上限。
4. 在每轮迭代中，首先计算相邻线程的编号，同时进行一些边界判断。
5. 接着进行异步的数据交换，接受数据一旦完成即可进行后续的操作。
6. 对于本进程的数据和相邻进程的数据进行归并，这里只存储本线程需要的部分。
7. 等待发送数据的完成，然后进行下一轮迭代。
8. 释放占用的内存。

## 性能优化

### 初版实现

```cpp
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "worker.h"

template<typename T>
void merge(T* a1, size_t len1, T* a2, size_t len2, T* buffer) {
	size_t len = len1 + len2, p1 = 0, p2 = 0;
	for (size_t i = 0; i < len; i++) {
		if (p1 == len1) buffer[i] = a2[p2++];
		else if (p2 == len2) buffer[i] = a1[p1++];
		else buffer[i] = a1[p1] < a2[p2] ? a1[p1++] : a2[p2++];
	}
}

void Worker::sort() {
	if (out_of_range)
		return;
	int round = 0;
	std::sort(data, data + block_len);
	while (++round <= nprocs) {
		size_t block_size = ceiling(n, nprocs);
		bool isSender = (round % 2) == (rank % 2);
		if (isSender) {
			int dest = rank + 1;
			if (last_rank)
				continue;
			MPI_Send(data, block_len, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
			MPI_Recv(data, block_len, MPI_FLOAT, dest, 0, MPI_COMM_WORLD, nullptr);
		} else {
			int source = rank - 1;
			if (source < 0)
				continue;
		  float *dataFromNeighbor = new float[block_size];
		  float *merged = new float[block_size + block_size];
			MPI_Status status;
			MPI_Recv(dataFromNeighbor, block_size, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
			int neighborLen;
			MPI_Get_count(&status, MPI_FLOAT, &neighborLen);
			merge(data, block_len, dataFromNeighbor, neighborLen, merged);
			size_t myOffset = 0, neighborOffset = 0;
			if (source > rank) neighborOffset += block_len;
			else myOffset += neighborLen;
			std::memcpy(data, merged + myOffset, sizeof(float) * block_len);
			MPI_Send(merged + neighborOffset, neighborLen, MPI_FLOAT, source, 0, MPI_COMM_WORLD);
	    delete[] dataFromNeighbor;
	    delete[] merged;	
    }
	}
}
```

### 优化策略

相比最初的实现，我的优化主要改进了下面几个部分：

- 将每轮归并所需的内存分配提前到迭代外，减少内存分配与释放的开销。
- 改进算法，抛弃一个线程承担主要计算的部分，让两个线程都进行计算，增加可并行性。
- 使用异步通信，将一半线程的通讯阶段和计算阶段重合。
- 改进归并过程，只进行本线程所需数据的归并。
- 减少内存的移动，直接交换归并后的指针。

### 优化效果

经过简单的测试后，在 $40$ 个进程下运行  $10^8$ 的数据由原本的平均 $1.3$ 秒提速至了 $0.75$ 秒，大约有 $1.7$ 倍的速度提升。

## 运行时间及加速比

对于每次测试，我们运行三次取最短的时间。

|    进程数    |  时间（毫秒）  |  加速比  |
| :----------: | :------------: | :------: |
| $1\times 1$  | $12688.997000$ | $1.000$  |
| $1\times 2$  | $6699.685000$  | $1.894$  |
| $1\times 4$  | $3575.101000$  | $3.549$  |
| $1\times 8$  | $1992.702000$  | $6.368$  |
| $1\times 16$ | $1225.265000$  | $10.356$ |
| $2\times 16$ |  $790.037000$  | $16.061$ |

  

