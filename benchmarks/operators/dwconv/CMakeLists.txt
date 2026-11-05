foreach(benchmark IN ITEMS dwconv_bench dwconv_fp16_bench dwconv_weights_bench dwconv_opt_bench)
    add_executable(${benchmark} ${benchmark}.cu)
    target_link_libraries(${benchmark} PRIVATE cuda_lab::dwconv)
endforeach()

add_executable(dwconv_bias_bench dwconv_bias_bench.cu)
target_include_directories(dwconv_bias_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dwconv_bias_bench PRIVATE CUDA::cudart)
