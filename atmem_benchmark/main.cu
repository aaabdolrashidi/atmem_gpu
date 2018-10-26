#include <stdio.h>
#include <stdint.h>

#include "kernel.cu"

int main(int argc, char* argv[])
{
	float *in_h;
	float *in_d;
	unsigned int num_elements, memory_block_size, thread_block_size;
	int mode, cache_warmup_en;

	num_elements = 1024 * 1024;

	memory_block_size = 64;
	thread_block_size = 512;
	mode = 0;
	cache_warmup_en = 0;

	if (argc == 6) {
		num_elements = atoi(argv[1]);
		thread_block_size = atoi(argv[2]);
		memory_block_size = atoi(argv[3]);
		mode = atoi(argv[4]);
		cache_warmup_en = atoi(argv[5]);
		if (mode < 0 || mode > 7)
		{
			printf("ERROR: Mode can only be an integer within [0, 7]!");
			exit(0);
		}
	}
	else {
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./atmem_bench [num_elements] [thread_block_size] [memory_block_size] [mode=0..7] [cache_warmup_en=0..1])"
			"\n");
		exit(0);
	}
	// Print all parameters
	printf("Number of elements = %u\nThread Block size = %u\nMemory Block size (Interval in Mode 5) = %u\nCache warmup = %s\nMode = %d (%s)\n",
		num_elements, thread_block_size, memory_block_size, 
		(cache_warmup_en == 0) ? "DISABLED" : "ENABLED",
		mode,
		(mode == 0) ? "BASELINE" :
		(mode == 1) ? "ATOMIC" :
		(mode == 2) ? "1 THREAD TO 1 ELEMENT" :
		(mode == 3) ? "1 WARP TO 1 ELEMENT" :
		(mode == 4) ? "1 WARP TO 32 ELEMENTS" :
		(mode == 5) ? "1 WARP TO 32 FAR ELEMENTS" :
		(mode == 6) ? "1 VECTOR ATOMIC ADD" :
		(mode == 7) ? "CLOCK64() FUNCTION OVERHEAD" :
		"ERROR");

	// Host array
	in_h = (float*)malloc(num_elements*sizeof(float));
	for (int i = 0; i<num_elements; i++)
	{
		in_h[i] = (float)(rand() % 1000) / 100.0;
	}

	// Print input
	#if IS_ARRAY_PRINT_ENABLED == 1
		printf("Input: ");
		for (int i = 0; i<num_elements; i++)
		{
		printf("%f ", in_h[i]);
		}
		printf("\n");
	#endif

	printf("Allocating device variables...\n");

	cudaMalloc((void**)&in_d, num_elements * sizeof(float));
	cudaDeviceSynchronize();

	// H to D
	printf("Copying data from host to device...\n");
	cudaMemcpy(in_d, in_h, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// Kernel Launch
	printf("Launching kernel...\n");

	atmem_bench(in_d, num_elements, thread_block_size, memory_block_size, mode, cache_warmup_en);

	// D to H
	printf("Copying data from device to host...\n");
	cudaMemcpy(in_h, in_d, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Print output
	#if IS_ARRAY_PRINT_ENABLED == 1
		printf("Output: ");
		for (int i = 0; i<num_elements; i++)
		{
			printf("%f ", in_h[i]);
		}
		printf("\n");
	#endif
	
	// Free memory
	cudaFree(in_d);
	free(in_h);

	return 0;
}
