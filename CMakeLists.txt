cmake_minimum_required(VERSION 4.0)
project(parallel_and_distributed_computations CUDA)

set(CMAKE_CUDA_STANDARD 20)

#set(CMAKE_CUDA_ARCHITECTURES "native")

include_directories(cuda_classes)
set(cudaClasses cuda_classes/array.cu cuda_classes/texture.cu
        cuda_classes/linear.cu)

add_executable(lab1 lab1/main.cu)

add_executable(lab2 lab2/main.cu
        ${cudaClasses})

add_executable(lab3 lab3/main.cu
        ${cudaClasses}
        lab3/pixel.cu)

#set_target_properties(lab1 PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(lab2 PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
