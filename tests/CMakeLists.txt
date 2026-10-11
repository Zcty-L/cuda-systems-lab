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
add_test(NAME dwconv_test COMMAND dwconv_test)
add_test(NAME dwconv_prefetch_test COMMAND dwconv_test --prefetch-only)

add_executable(dwconv_fp16_test dwconv_fp16_test.cu)
target_link_libraries(dwconv_fp16_test PRIVATE cuda_ops_dwconv CUDA::cudart)
add_test(NAME dwconv_fp16_test COMMAND dwconv_fp16_test)
