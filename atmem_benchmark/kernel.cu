typedef signed long long TimeType;
/******************************************************************************
* Host Functions
*******************************************************************************/

TimeType find_max(TimeType* time_array, int length)
{
	TimeType result = time_array[0];
	for (int i = 1; i < length; i++)
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
	if (threadIdx.x == 0) start_time = clock64();
	__syncthreads();
	// Begin	
	int index = (blockIdx.x*blockDim.x + threadIdx.x)*blockSize;
	for (int i = 0; i<blockSize; i++) {
		data[index + i] += scalar;
	}
	// End
	__syncthreads();
	if (threadIdx.x == 0)
	{
		end_time = clock64();
		temp = end_time - start_time;
		elapsed_time[blockIdx.x] = temp;
		// printf("Elapsed time: %u\n", temp);
	}
	__threadfence();
}

__global__ void lmwTest_atomic(float* data, float scalar, int blockSize, TimeType* elapsed_time) {
	TimeType start_time, end_time, temp;
	if (threadIdx.x == 0) start_time = clock64();
	__syncthreads();
	// Begin	
	int index = (blockIdx.x*blockDim.x + threadIdx.x)*blockSize;
	for (int i = 0; i<blockSize; i++) {
		atomicAdd(&(data[index + i]), scalar);
	}
	// End
	__syncthreads();
	if (threadIdx.x == 0)
	{
		end_time = clock64();
		temp = end_time - start_time;
		elapsed_time[blockIdx.x] = temp;
		// printf("Elapsed time: %lld\n", temp);
	}
	__threadfence();
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
	int elapsed_time_size = num_blocks;
	cudaError cuda_status;

	elapsed_time_h = (TimeType*)malloc(elapsed_time_size * sizeof(TimeType));
	for (int i = 0; i < elapsed_time_size; i++)
		elapsed_time_h[i] = 3;	// Non-zero random value

	cudaMalloc((void**)&elapsed_time_d, elapsed_time_size * sizeof(TimeType));
	cudaMemset(elapsed_time_d, 0, elapsed_time_size * sizeof(TimeType));
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
	cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float total_elapsed_time = 0;
	cudaEventElapsedTime(&total_elapsed_time, start, stop);

	cudaDeviceSynchronize();

	// Copying time to host
	cuda_status = cudaMemcpy(elapsed_time_h, elapsed_time_d, elapsed_time_size * sizeof(TimeType), cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess)
	{
		printf("\n*CUDA Error %d during cudaMemcpy(D -> H)!*\n\n", cuda_status);
	}
	cudaDeviceSynchronize();
	//for (int i = 0; i < num_blocks; i++)
	//	printf("%d\n", elapsed_time_h[i]);

	printf("Total elapsed kernel time %f ms\n", total_elapsed_time);
	printf("Max in-SM cycles: %d cycles\n", find_max(elapsed_time_h, elapsed_time_size));

	free(elapsed_time_h);
	cudaFree(elapsed_time_d);
}
