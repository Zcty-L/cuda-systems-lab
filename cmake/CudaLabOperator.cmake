
function(cuda_lab_add_operator name)
    cmake_parse_arguments(OP "" "" "SOURCES" ${ARGN})
    if(NOT OP_SOURCES OR OP_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "cuda_lab_add_operator(${name}) 需要 SOURCES 参数")
    endif()

    add_library(cuda_ops_${name} STATIC ${OP_SOURCES})
    add_library(cuda_lab::${name} ALIAS cuda_ops_${name})
    set_target_properties(cuda_ops_${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(cuda_ops_${name} PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(cuda_ops_${name} PUBLIC CUDA::cudart)
endfunction()
