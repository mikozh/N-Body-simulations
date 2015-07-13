CFLAGS= -O3
OBJ  = nbody_Main.o nbody_CUDA.o
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcuda

%.o: %.c
	icc -c -o $@ $< $(CFLAGS) -openmp
%.o: %.cu
	nvcc -c -o $@ $< -arch=sm_20 $(CFLAGS) $(LIBS)

hybr: $(OBJ)
	icc -o $@ $^ $(CFLAGS) -openmp $(LIBS)

clean: 
	rm -f *.o hybr
