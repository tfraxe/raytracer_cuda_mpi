raytracer.o: raytracer.cu
	nvcc -c -Wno-deprecated-gpu-targets -m64 -gencode arch=compute_50,code=sm_50 raytracer.cu

main.o: main.cpp
	OMPI_CC=nvcc mpic++ -c main.cpp -lcudart -L /usr/local/cuda/lib64

main: main.o raytracer.o
	mpic++ main.o raytracer.o -lcudart -L /usr/local/cuda/lib64

clean:
	rm -rf *.o
