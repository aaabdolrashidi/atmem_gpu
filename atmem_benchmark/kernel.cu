typedef unsigned long long TimeType;
/******************************************************************************
* Host Functions
*******************************************************************************/

TimeType find_max(TimeType* time_array, int length)
{
	TimeType result = 0;
	for (int i = 0; i < length; i++)
	{
		if (result < time_array[i]) result = time_array[i];
	}
	return result;
}

/******************************************************************************
* Kernels
*******************************************************************************/
__global__ void lmwTest_baseline(float* data, float scalar, int blockSize, TimeType* elapsed_time) {
	TimeType start_time, end_time, temp;
	start_time = clock64();
	// Begin	
	int index = (blockIdx.x*blockDim.x + threadIdx.x)*blockSize;
	for (int i = 0; i<blockSize; i++) {
		data[index + i] += scalar;
	}
	// End
	end_time = clock64();
	temp = end_time - start_time;
	elapsed_time[blockIdx.x] = temp;
	// printf("Elapsed time: %u\n", temp);
}

__global__ void lmwTest_atomic(float* data, float scalar, int blockSize, TimeType* elapsed_time) {
	TimeType start_time, end_time, temp;
	start_time = clock64();
	// Begin	
	int index = (blockIdx.x*blockDim.x + threadIdx.x)*blockSize;
	for (int i = 0; i<blockSize; i++) {
		atomicAdd(&(data[index + i]), scalar);
	}
	// End
	end_time = clock64();
	temp = end_time - start_time;
	elapsed_time[blockIdx.x] = temp;
	printf("Elapsed time: %u\n", temp);
}

/******************************************************************************
* End of Kernel Function Definitions; proceeding to the invocation section
*******************************************************************************/

void atmem_bench(float* input, unsigned int num_elements, unsigned int block_size, int mode = 0) {
	int num_blocks = (num_elements - 1) / block_size + 1;
	printf("Number of blocks: %d\n", num_blocks);
	// Setting up time parameters
	TimeType* elapsed_time_d;
	TimeType* elapsed_time_h;
	elapsed_time_h = (TimeType*)malloc(num_blocks * sizeof(TimeType));
	for (int i = 0; i < num_blocks; i++)
	{
		elapsed_time_h[i] = 0;
	}

	cudaMalloc((void**)&elapsed_time_d, num_blocks * sizeof(TimeType));
	cudaDeviceSynchronize();
	cudaMemcpy(elapsed_time_d, elapsed_time_h, num_blocks * sizeof(TimeType), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	// Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Invoking Kernel
	cudaEventRecord(start);
	if (mode == 0)
		lmwTest_baseline << < num_blocks, block_size >> > (input, 1.0, block_size, elapsed_time_d);
	else if (mode == 1)
		lmwTest_atomic << < num_blocks, block_size >> > (input, 1.0, block_size, elapsed_time_d);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float total_elapsed_time = 0;
	cudaEventElapsedTime(&total_elapsed_time, start, stop);

	cudaDeviceSynchronize();

	// Copying time to host
	cudaMemcpy(elapsed_time_h, elapsed_time_d, num_blocks * sizeof(TimeType), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Total elapsed kernel time %f ms\n", total_elapsed_time);
	printf("Max in-SM cycles: %u cycles\n", find_max(elapsed_time_h, num_blocks));

	free(elapsed_time_h);
	cudaFree(elapsed_time_d);
}
