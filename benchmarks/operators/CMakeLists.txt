add_executable(vector_add_bench vector_add_bench.cu)
target_include_directories(
    vector_add_bench
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/ops/vector_add
)
target_link_libraries(vector_add_bench PRIVATE cuda_ops_vector_add CUDA::cudart)

add_executable(softmax_bench softmax_bench.cu)
target_link_libraries(softmax_bench PRIVATE cuda_ops_softmax CUDA::cudart)

add_executable(softmax_cluster_bench softmax_cluster_bench.cu)
target_link_libraries(softmax_cluster_bench PRIVATE cuda_ops_softmax CUDA::cudart)

add_executable(dwconv_bench dwconv_bench.cu)
target_link_libraries(dwconv_bench PRIVATE cuda_ops_dwconv CUDA::cudart)

add_executable(dwconv_fp16_bench dwconv_fp16_bench.cu)
target_link_libraries(dwconv_fp16_bench PRIVATE cuda_ops_dwconv CUDA::cudart)

add_executable(dwconv_weights_bench dwconv_weights_bench.cu)
target_link_libraries(dwconv_weights_bench PRIVATE cuda_ops_dwconv CUDA::cudart)
