cmake_minimum_required(VERSION 3.22)

#project(DBSCAN_Parallel CUDA)
project(DBSCAN_Parallel LANGUAGES CUDA CXX)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)

add_executable(DBSCAN_Parallel main.cu
        SequentialDBSCAN.cpp
        SequentialDBSCAN.h
        Points.cpp
        Points.h)

set_target_properties(DBSCAN_Parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
