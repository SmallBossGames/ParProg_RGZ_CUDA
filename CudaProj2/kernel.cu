#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <forward_list>
#include <chrono>

cudaError_t findSimpleDividersWithCUDA(std::forward_list<long long> *result, long long value, int cudaCores);

__device__ bool isPrime(long long value)
{
	for (int i = 2; i <= sqrt((double) value); i++)
	{
		if (value%i == 0)
			return false;
	}
	return true;
}

__global__ void getSimpleDividersKernel(char *output, long long from, long long value, int step)
{
	const long long currentDefault = threadIdx.x + from;

	for (long long i = currentDefault; i <= value; i+=step)
	{
		long long tempVal = value;
		long long outPos = i - from;

		output[outPos] = 0;

		if (tempVal%i == 0 && isPrime(i))
		{
			while (tempVal%i == 0)
			{
				output[outPos]++;
				tempVal /= i;
			}
		}
	}
}


int main()
{
	using namespace std;

	char *outputArray = nullptr;
	long long value;
	std::forward_list<long long> result;

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
		std::cout << (long long) item << ' ';
	}

    return 0;
}

cudaError_t findSimpleDividersWithCUDA(std::forward_list<long long> *result, long long value, int cudaCores)
{
	const long long from = 2;
	const long long buferSize = value - from;

	if (buferSize < cudaCores)
	{
		cudaCores = buferSize;
	}

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


	getSimpleDividersKernel <<<1, cudaCores >>> (dev_output, from, value, cudaCores);

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

	for (long long j = 0; j < buferSize; j++)
	{
		int itCount = (int)buffer_output[j];
		for (int k = 0; k < itCount; k++)
		{
			long long tempValue = from + j;
			result->push_front(tempValue);
		}
	}

Error:
	delete[] buffer_output;
	cudaFree(dev_output);

	return cudaStatus;
}
