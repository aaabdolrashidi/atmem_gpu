#include <stdio.h>
#include <stdint.h>

#include "kernel.cu"

int main(int argc, char* argv[])
{
	float *in_h;
	float *in_d;
	unsigned int num_elements, memory_block_size, thread_block_size;
	int mode;

	num_elements = 1024*1024;
	
	memory_block_size = 64;
	thread_block_size = 512;
	mode = 0;

/*	if (argc == 2) {
		num_elements = atoi(argv[1]);
	}
	else if (argc == 3) {
		num_elements = atoi(argv[1]);
		thread_block_size = atoi(argv[2]);
	}
	else if (argc == 4) {
		num_elements = atoi(argv[1]);
		thread_block_size = atoi(argv[2]);
		memory_block_size = atoi(argv[3]);
	}
	else*/ if (argc == 5) {
		num_elements = atoi(argv[1]);
		thread_block_size = atoi(argv[2]);
		memory_block_size = atoi(argv[3]);
		mode = atoi(argv[4]);
		if (mode != 0 && mode != 1)
		{
			printf("ERROR: Mode can only be an integer within [0, 1]!");
			exit(0);
		}
	}
	else {
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./atmem_bench [num_elements] [thread_block_size] [memory_block_size] [mode=0|1])"
			"\n");
		exit(0);
	}
	// Print all parameters
	printf("Number of elements = %u\nThread Block size = %u\nMemory Block size = %u\nMode = %s (%d)\n", num_elements, thread_block_size, memory_block_size, (mode == 0) ? "BASELINE" : "ATOMIC", mode);

	// Host array
	in_h = (float*)malloc(num_elements*sizeof(float));
	for (int i = 0; i<num_elements; i++)
	{
		in_h[i] = (float)(rand() % 1000) / 100.0;
	}

	// Print input
	//printf("Input: ");
	//for (int i = 0; i<num_elements; i++)
	//{
	//printf("%f ", in_h[i]);
	//}
	//printf("\n");
	
	printf("Allocating device variables...\n");

	cudaMalloc((void**)&in_d, num_elements * sizeof(float));
	cudaDeviceSynchronize();

	// H to D
	printf("Copying data from host to device...\n");
	cudaMemcpy(in_d, in_h, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// Kernel Launch
	printf("Launching kernel...\n");

	atmem_bench(in_d, num_elements, thread_block_size, memory_block_size, mode);

	// D to H
	printf("Copying data from device to host...\n");
	cudaMemcpy(in_h, in_d, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// Print output
	//printf("Output: ");
	//for (int i = 0; i<num_elements; i++)
	//{
	//	printf("%f ", in_h[i]);
	//}
	//printf("\n");
	
	cudaFree(in_d);
	free(in_h);

	return 0;
}
