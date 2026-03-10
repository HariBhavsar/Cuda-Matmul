all:
	nvcc -arch=sm_89 -O3 partA.cu -lcublas -o bins/partA.bin

clean:
	rm partA