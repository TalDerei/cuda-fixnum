# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# compile CUDA with /usr/local/cuda-11.7/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_FLAGS =  -Xcompiler -Wall,-Wextra --gpu-code=sm_60 --gpu-architecture=compute_60 --relocatable-device-code=true  

CUDA_DEFINES = -DCURVE_ -DMULTICORE=1 -DNO_PROCPS -DNO_PT_COMPRESSION=1

CUDA_INCLUDES = -I/home/ubuntu/cuda-fixnum/curve-operations/. -I/home/ubuntu/cuda-fixnum/curve-operations/./src -I/home/ubuntu/cuda-fixnum/curve-operations/./cuda-fixnum 

CXX_FLAGS =  -std=c++14 -Wall -Wextra -Wfatal-errors -fopenmp -ggdb3 -O2 -march=westmere -mtune=skylake-avx512  

CXX_DEFINES = -DCURVE_ -DMULTICORE=1 -DNO_PROCPS -DNO_PT_COMPRESSION=1

CXX_INCLUDES = -I/home/ubuntu/cuda-fixnum/curve-operations/. -I/home/ubuntu/cuda-fixnum/curve-operations/./src -I/home/ubuntu/cuda-fixnum/curve-operations/./cuda-fixnum 

