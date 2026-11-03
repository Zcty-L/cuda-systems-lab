
# example of a native test for the vector_add operation
if(TARGET cuda_ops_vector_add)
    add_executable(vector_add_test vector_add_test.cu)
    target_link_libraries(vector_add_test PRIVATE cuda_lab::vector_add)
    add_test(NAME vector_add_test COMMAND vector_add_test)
    set_tests_properties(vector_add_test PROPERTIES LABELS "native;cuda;vector_add")
endif()
