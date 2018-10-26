
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_35
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = atmem_bench
OBJ	        = main.o

default: $(EXE)

main.o: main.cu kernel.cu
	$(NVCC) -G -c -o $@ main.cu $(NVCC_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

ptx:
	$(NVCC) -ptx kernel.cu

clean:
	rm -rf *.o $(EXE)
