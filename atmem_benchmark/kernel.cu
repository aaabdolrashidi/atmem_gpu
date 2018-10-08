#define BLOCK_SIZE 64

__global__ void lmwTest_baseline(float* data, float scalar, int blockSize){
//	long long int startTime, endTime;
//	startTime = clock64();

	// Begin	
	int index = (blockIdx.x*blockDim.x+threadIdx.x)*blockSize;
	for (int i=0; i<blockSize; i++){
		data[index+i] += scalar; 
	}
	// End
//	endTime = clock64();
//	printf("Elapsed time: %ll", endTime - startTime);
}

__global__ void lmwTest_atomic(float* data, float scalar, int blockSize){
	int index = (blockIdx.x*blockDim.x+threadIdx.x)*blockSize;
	for (int i=0; i<blockSize; i++){
		atomicAdd(&(data[index+i]), scalar);
	}
}

/******************************************************************************
 * End of Kernel Function Definitions; proceeding to the invocation section
*******************************************************************************/

void atmem_bench(float* input, unsigned int num_elements) {
	int numOfBlocks=(num_elements-1) / BLOCK_SIZE + 1;
	lmwTest_baseline<<<numOfBlocks,BLOCK_SIZE>>>(input,1.0,BLOCK_SIZE);

        cudaDeviceSynchronize();
}


