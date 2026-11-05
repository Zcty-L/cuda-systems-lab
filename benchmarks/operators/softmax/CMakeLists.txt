add_executable(softmax_bench softmax_bench.cu)
target_link_libraries(softmax_bench PRIVATE cuda_lab::softmax)

add_executable(softmax_cluster_bench softmax_cluster_bench.cu)
target_link_libraries(softmax_cluster_bench PRIVATE cuda_lab::softmax)
