CXX=g++
CXXFLAGS=-O2 -Wall -fopenmp
NVCC=nvcc
NVCCFLAGS=-O2

all: naive order threads target cuda

naive:
	$(CXX) $(CXXFLAGS) -o naive naive.cpp

order:
	$(CXX) $(CXXFLAGS) -o order order.cpp

threads:
	$(CXX) $(CXXFLAGS) -o threads threads.cpp

target:
	$(CXX) $(CXXFLAGS) -o target target.cpp

cuda:
	$(NVCC) $(NVCCFLAGS) -o cuda cuda.cu

.PHONY: clean run test

clean:
	rm -f naive order factorized threads target cuda
	rm -f time.*.txt
