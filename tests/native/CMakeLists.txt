
# example of a native test for the vector_add operation
if(TARGET cuda_ops_vector_add)
    add_executable(vector_add_test vector_add_test.cu)
    target_link_libraries(vector_add_test PRIVATE cuda_lab::vector_add)
    add_test(NAME vector_add_test COMMAND vector_add_test)
    set_tests_properties(vector_add_test PROPERTIES LABELS "native;cuda;vector_add")
endif()

if(TARGET cuda_ops_softmax)
    add_executable(softmax_test softmax_test.cu)
    target_link_libraries(softmax_test PRIVATE cuda_lab::softmax)
    add_test(NAME softmax_test COMMAND softmax_test)
    set_tests_properties(softmax_test PROPERTIES LABELS "native;cuda;softmax")
endif()

if(TARGET cuda_ops_dwconv)
    add_executable(dwconv_test dwconv_test.cu)
    target_link_libraries(dwconv_test PRIVATE cuda_lab::dwconv)
    if(TARGET cuda_ops_dwconv_cudnn)
        target_link_libraries(dwconv_test PRIVATE cuda_lab::dwconv_cudnn)
        target_compile_definitions(dwconv_test PRIVATE DWCONV_HAS_CUDNN=1)
        add_test(NAME dwconv_cudnn_nhwc_test COMMAND dwconv_test --quick --kernel cudnn --layout nhwc --dtype all --check-cpu --warmup 0 --iterations 1)
        set_tests_properties(dwconv_cudnn_nhwc_test PROPERTIES LABELS "native;cuda;dwconv;cudnn")
    endif()
    # 手动运行默认只测速；CTest 显式保留所有实现的 CPU 正确性校验。
    add_test(NAME dwconv_test COMMAND dwconv_test --quick --kernel all --dtype all --check-cpu --warmup 0 --iterations 1)
    set_tests_properties(dwconv_test PROPERTIES LABELS "native;cuda;dwconv")
endif()
