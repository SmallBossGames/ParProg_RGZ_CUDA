#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <forward_list>
#include <chrono>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t findSimpleDividersWithCUDA(std::forward_list<long> *result, long long value, int cudaCores);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


__device__ bool isPrime(long long value)
{
	for (int i = 2; i <= sqrt((double) value); i++)
	{
		if (value%i == 0)
			return false;
	}
	return true;
}

__global__ void getSimpleDividersKernel(char *output, long long from, long long value)
{
	cudaError_t status;

	unsigned int i = threadIdx.x;
	long long current = i + from;

	output[i] = 0;

	if (value%current != 0 || !isPrime(current))
	{
		return;
	}

	while (value%current == 0)
	{
		output[i]++;
		value /= current;
	}
}


int main()
{
	using namespace std;

	char *outputArray = nullptr;
	int value;
	std::forward_list<long> result;

	std::cout << "Write value:" << std::endl;
	std::cin >> value;

	auto begin = chrono::high_resolution_clock::now();

    // Add vectors in parallel.
	cudaError_t cudaStatus = findSimpleDividersWithCUDA(&result, value, 1000);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "findSimpleDividersWithCUDA failed!");
        return 1;
    }

	auto end = chrono::high_resolution_clock::now();

	auto run_time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();

	cout << "It's done in " << run_time << "ms" << endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	for (auto item : result)
	{
		std::cout << (int) item << ' ';
	}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t findSimpleDividersWithCUDA(std::forward_list<long> *result, long long value, int cudaCores)
{
	const int from = 2;
	const int buferSize = value > cudaCores ? cudaCores : value - from;
	const int divCount = (value - from) / buferSize + ((value-from)%buferSize == 0 ? 0 : 1 );

	char *buffer_output = new char[buferSize];
	char *dev_output;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output).

	cudaStatus = cudaMalloc(&dev_output, buferSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (int i = 0; i < divCount; i++)
	{
		long long start = i * buferSize + from;
		int taskCount = (i == divCount - 1) ? value - start : buferSize;
		
		
		getSimpleDividersKernel << <1, taskCount >> > (dev_output, start, value);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		cudaStatus = cudaMemcpy(buffer_output, dev_output, buferSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy 260 failed!");
			goto Error;
		}

		for (int j = 0; j < taskCount; j++)
		{
			int itCount = (int)buffer_output[j];
			for (int k = 0; k < itCount; k++)
			{
				long long tempValue = start + j;
				result->push_front(tempValue);
			}
		}
	}

Error:
	delete[] buffer_output;
	cudaFree(dev_output);

	return cudaStatus;
}
