add_library(cuda_ops_vector_add STATIC vector_add.cu)

target_include_directories(
    cuda_ops_vector_add
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(cuda_ops_vector_add PUBLIC CUDA::cudart)
