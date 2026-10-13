add_library(cuda_ops_dwconv STATIC dwconv_pytorch.cu dwconv.cu dwconv_fp16.cu dwconv_opt_fp32.cu dwconv_opt_fp16.cu)
target_include_directories(cuda_ops_dwconv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cuda_ops_dwconv PUBLIC CUDA::cudart)

option(DWCONV_TEST_CUDNN "在 dwconv_test 中启用 cuDNN 对比" ON)
if(DWCONV_TEST_CUDNN)
    find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
    find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        add_library(cuda_ops_dwconv_cudnn STATIC dwconv_cudnn.cu)
        target_include_directories(cuda_ops_dwconv_cudnn
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
            PRIVATE ${CUDNN_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(cuda_ops_dwconv_cudnn PUBLIC CUDA::cudart PRIVATE ${CUDNN_LIBRARY})
    else()
        message(STATUS "未找到 cuDNN，dwconv_test 禁用 cudnn 选项")
    endif()
endif()
