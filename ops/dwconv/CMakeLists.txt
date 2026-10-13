add_library(cuda_ops_dwconv STATIC dwconv_pytorch.cu dwconv.cu dwconv_fp16.cu dwconv_opt_fp32.cu dwconv_opt_fp16.cu)
target_include_directories(cuda_ops_dwconv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cuda_ops_dwconv PUBLIC CUDA::cudart)
