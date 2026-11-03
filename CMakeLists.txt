cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_systems_lab VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
include(CTest)

option(CUDA_LAB_BUILD_PYTHON "构建 PyTorch 接口和 Python 包" OFF)
option(CUDA_LAB_BUILD_BENCHMARKS "构建原生性能基准" ON)
option(CUDA_LAB_BUILD_TOOLS "构建设备查询等工具" ON)
option(CUDA_LAB_BUILD_STUDIES "构建已注册的架构学习实验" OFF)
set(CUDA_LAB_OPS "vector_add" CACHE STRING "本次构建的算子，以分号分隔；空值仅构建工具")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CudaLabOperator)
add_subdirectory(ops)

if(CUDA_LAB_BUILD_PYTHON)
    add_subdirectory(bindings/torch)
endif()

if(CUDA_LAB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/operators)
endif()

if(CUDA_LAB_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(CUDA_LAB_BUILD_STUDIES)
    add_subdirectory(studies)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
