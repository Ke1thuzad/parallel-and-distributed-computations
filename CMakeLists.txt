cmake_minimum_required(VERSION 4.0)
project(parallel_and_distributed_computations CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(lab1 lab1/main.cu)

set_target_properties(lab1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#add_executable(hw1 hw1/main.cpp)
#add_executable(hw2 hw2/hw2.cpp)
