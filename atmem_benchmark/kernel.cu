#define IS_ARRAY_PRINT_ENABLED 0

const int reps = 512;
#define REPEAT_FUNC REPEAT512
#define REPEAT64(N) N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N
#define REPEAT128(N) N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N
#define REPEAT256(N) N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N
#define REPEAT512(N) N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N N

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
// Cache warm-up
__global__ void cacheWarmup(float* data, int length) {
	volatile int index = blockIdx.x*blockDim.x + threadIdx.x;
	volatile float temp = 0;
	if (index < length)
	{
		temp = data[index];
	}
}
__device__ void cacheWarmup_Device(float* data, int length) {
	volatile int index = blockIdx.x*blockDim.x + threadIdx.x;
	volatile float temp = 0;
	if (index < length)
	{
		temp = data[index];
	}
}

// Mode 0
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

// Mode 1
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

// Mode 2: One thread performs one atomic add on one element
__global__ void oneThreadOneVarAtomicAdd(float* data, float scalar, TimeType* elapsed_time) {
	TimeType start_time, end_time, temp;
	start_time = clock64();
	// Begin
	REPEAT_FUNC(atomicAdd(&(data[0]), scalar); __threadfence();)
	// End
	end_time = clock64();
	temp = (end_time - start_time) / reps;
	elapsed_time[blockIdx.x] = temp;
	// printf("Start: %lld\nEnd: %lld\n", start_time, end_time);
	// printf("Elapsed time: %lld\n", temp);
}

// Mode 3: All threads in a warp perform one atomic add on one element
__global__ void oneWarpOneVarAtomicAdd(float* data, float scalar, TimeType* elapsed_time) {
	cacheWarmup_Device(data, 32);

	TimeType start_time, end_time, temp;
	if (threadIdx.x == 0) start_time = clock64();
	__syncthreads();
	// Begin	
	REPEAT_FUNC(atomicAdd(&(data[0]), scalar); __threadfence();)
	// End
	__syncthreads();
	if (threadIdx.x == 0)
	{
		end_time = clock64();
		temp = (end_time - start_time) / reps;
		elapsed_time[blockIdx.x] = temp;
		// printf("Elapsed time: %lld\n", temp);
	}
}

// Mode 4: Every thread in a warp performs one atomic add on one element (i.e. 32 elements in total)
__global__ void oneWarp32VarAtomicAdd(float* data, float scalar, TimeType* elapsed_time) {
	cacheWarmup_Device(data, 32);

	TimeType start_time, end_time, temp;
	if (threadIdx.x == 0) start_time = clock64();
	__syncthreads();
	// Begin	
	REPEAT_FUNC(atomicAdd(&data[threadIdx.x], scalar); __threadfence();)
	// End
	__syncthreads();
	if (threadIdx.x == 0)
	{
		end_time = clock64();
		temp = (end_time - start_time) / reps;
		elapsed_time[blockIdx.x] = temp;
		// printf("Elapsed time: %lld\n", temp);
	}
}

// Mode 5: Like Mode 4, but the elements are far from one another.
__global__ void oneWarp32VarAtomicAdd_Far(float* data, float scalar, int interval, TimeType* elapsed_time) {
	cacheWarmup_Device(data, 32);

	TimeType start_time, end_time, temp;
	if (threadIdx.x == 0) start_time = clock64();
	__syncthreads();
	// Begin	
	REPEAT_FUNC(atomicAdd(&data[interval*threadIdx.x], scalar); __threadfence();)
	// End
	__syncthreads();
	if (threadIdx.x == 0)
	{
		end_time = clock64();
		temp = (end_time - start_time) / reps;
		elapsed_time[blockIdx.x] = temp;
		// printf("Elapsed time: %lld\n", temp);
	}
	__threadfence();
}

// Mode 6: All elements in the vector perform one atomic add to a corresponding element.
__global__ void vectorAtomicAdd(float* data, float scalar, int length, TimeType* elapsed_time) {
	cacheWarmup_Device(data, 32);

	TimeType start_time, end_time, temp;
	if (threadIdx.x == 0) start_time = clock64();
	__syncthreads();
	// Begin	
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < length)
	{
		REPEAT_FUNC(atomicAdd(&data[index], scalar); __threadfence();)
	}
	// End
	__syncthreads();
	if (threadIdx.x == 0)
	{
		end_time = clock64();
		temp = (end_time - start_time) / reps;
		elapsed_time[blockIdx.x] = temp;
		// printf("Elapsed time: %lld\n", temp);
	}
}

// Mode 7: Clock64 overhead
__global__ void clock64_overhead(TimeType* elapsed_time) {
	TimeType start_time, end_time, temp;
	start_time = clock64();
	// Begin
	// ... Nothing!
	// End
	end_time = clock64();
	temp = (end_time - start_time);
	elapsed_time[blockIdx.x] = temp;
	// printf("Start: %lld\nEnd: %lld\n", start_time, end_time);
	// printf("Elapsed time: %lld\n", temp);
}
/******************************************************************************
* End of Kernel Function Definitions; proceeding to the invocation section
*******************************************************************************/

void atmem_bench(float* input, unsigned int num_elements, unsigned int thread_block_size, unsigned int memory_block_size, int mode = 0, int cache_warmup_en = 0) {
	int num_blocks;
	if(mode < 2) num_blocks = (num_elements / memory_block_size) / thread_block_size;
	else num_blocks = (num_elements - 1) / thread_block_size + 1;
	printf("Number of blocks: %d\n", num_blocks);
	// Setting up time parameters
	TimeType* elapsed_time_d;
	TimeType* elapsed_time_h;
	int elapsed_time_size = num_blocks;
	cudaError cuda_status;

	elapsed_time_h = (TimeType*)malloc(elapsed_time_size * sizeof(TimeType));
	for (int i = 0; i < elapsed_time_size; i++)
		elapsed_time_h[i] = -1;	// Non-zero random value

	cudaMalloc((void**)&elapsed_time_d, elapsed_time_size * sizeof(TimeType));
	cudaMemset(elapsed_time_d, 0, elapsed_time_size * sizeof(TimeType));
	cudaDeviceSynchronize();

	// Cache warm-up kernel
	if(cache_warmup_en) cacheWarmup << < num_blocks, thread_block_size >> > (input, num_elements);


	// Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Invoking Kernel
	cudaEventRecord(start);
	if (mode == 0)
		lmwTest_baseline << < num_blocks, thread_block_size >> > (input, 3.1, thread_block_size, elapsed_time_d);
	else if (mode == 1)
		lmwTest_atomic << < num_blocks, thread_block_size >> > (input, 3.1, thread_block_size, elapsed_time_d);
	else if (mode == 2)
		oneThreadOneVarAtomicAdd << < 1, 1 >> > (input, 3.1, elapsed_time_d);
	else if (mode == 3)
		oneWarpOneVarAtomicAdd << < 1, thread_block_size >> > (input, 3.1, elapsed_time_d);
	else if (mode == 4)
		oneWarp32VarAtomicAdd << < 1, thread_block_size >> > (input, 3.1, elapsed_time_d);
	else if (mode == 5)
		oneWarp32VarAtomicAdd_Far << < 1, thread_block_size >> > (input, 3.1, memory_block_size, elapsed_time_d);
	else if (mode == 6)
		vectorAtomicAdd << < num_blocks, thread_block_size >> > (input, 3.1, num_elements, elapsed_time_d);
	else if (mode == 7)
		clock64_overhead << <1, 1 >> > (elapsed_time_d);
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
		exit(1);
	}
	cudaDeviceSynchronize();
	//for (int i = 0; i < num_blocks; i++)
	//	printf("%d\n", elapsed_time_h[i]);

	printf("Total elapsed kernel time %f ms\n", total_elapsed_time);
	printf("Max in-SM cycles: %d cycles\n", find_max(elapsed_time_h, elapsed_time_size));

	free(elapsed_time_h);
	cudaFree(elapsed_time_d);
}
