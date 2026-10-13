add_executable(vector_add_test vector_add_test.cu)
target_include_directories(
    vector_add_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/ops/vector_add
)
target_link_libraries(vector_add_test PRIVATE cuda_ops_vector_add CUDA::cudart)
add_test(NAME vector_add_test COMMAND vector_add_test)

add_executable(softmax_test softmax_test.cu)
target_link_libraries(softmax_test PRIVATE cuda_ops_softmax CUDA::cudart)
add_test(NAME softmax_test COMMAND softmax_test)

add_executable(dwconv_test dwconv_test.cu)
target_link_libraries(dwconv_test PRIVATE cuda_ops_dwconv CUDA::cudart)
if(TARGET cuda_ops_dwconv_cudnn)
    target_link_libraries(dwconv_test PRIVATE cuda_ops_dwconv_cudnn)
    target_compile_definitions(dwconv_test PRIVATE DWCONV_HAS_CUDNN=1)
    add_test(NAME dwconv_cudnn_nhwc_test COMMAND dwconv_test --quick --kernel cudnn --layout nhwc --dtype all --check-cpu --warmup 0 --iterations 1)
endif()
# 手动运行默认只测速；CTest 显式保留所有实现的 CPU 正确性校验。
add_test(NAME dwconv_test COMMAND dwconv_test --quick --kernel all --dtype all --check-cpu --warmup 0 --iterations 1)
