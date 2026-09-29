cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_systems_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
include(CTest)

add_subdirectory(ops/vector_add)
add_subdirectory(ops/softmax)
add_subdirectory(benchmarks/operators)
add_subdirectory(benchmarks/memory)
add_subdirectory(tools)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
