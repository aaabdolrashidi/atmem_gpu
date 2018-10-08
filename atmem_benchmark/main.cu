#include <stdio.h>
#include <stdint.h>

#include "kernel.cu"

int main(int argc, char* argv[])
{
	float *in_h;
	float *in_d;
	unsigned int num_elements;
	unsigned int mode;

	if (argc == 1) {
		num_elements = 1000;
		mode = 0;
	}
	else if (argc == 2) {
		num_elements = atoi(argv[1]);
		mode = 0;
	}
	else if (argc == 3) {
		num_elements = atoi(argv[1]);
		mode = atoi(argv[2]);
		if (mode != 0 && mode != 1)
		{
			printf("ERROR: Mode can only be an integer within [0, 1]!");
			exit(0);
		}
	}
	else {
		printf("\n    Invalid input parameters!"
			"\n    Usage: ./atmem_bench            # Number of elements: 1,000\tMode: 0"
			"\n    Usage: ./atmem_bench <n>        # Number of elements: n\tMode: 0"
			"\n    Usage: ./atmem_bench <n> <m>    # Number of elements: n\tMode: m (0=Baseline, 1=Atomic)"
			"\n");
		exit(0);
	}

	// Host array
	in_h = (float*)malloc(num_elements*sizeof(float));

	printf("Input: ");
	for (int i = 0; i<num_elements; i++)
	{
		in_h[i] = (float)(rand() % 1000) / 100.0;
		printf("%f ", in_h[i]);
	}
	printf("\n");
	printf("Array size = %u\n", num_elements);

	printf("Allocating device variables...\n");

	cudaMalloc((void**)&in_d, num_elements * sizeof(float));
	cudaDeviceSynchronize();

	// H to D
	printf("Copying data from host to device...\n");
	cudaMemcpy(in_d, in_h, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// Kernel Launch
	printf("Launching kernel...\n");

	atmem_bench(in_d, num_elements);

	// D to H
	printf("Copying data from device to host...\n");
	cudaMemcpy(in_h, in_d, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Print output
	printf("Output: ");
	for (int i = 0; i<num_elements; i++)
	{
		printf("%f ", in_h[i]);
	}
	printf("\n");

	cudaFree(in_d);
	free(in_h);

	return 0;
}
