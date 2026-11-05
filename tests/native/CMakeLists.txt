# 按所选算子注册原生测试。
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
