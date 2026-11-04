set(memory_benchmarks
    smem_bandwidth smem_latency
    l1cache_bandwidth l1cache_latency
    l2cache_bandwidth l2cache_latency
    dram_bandwidth dram_latency
)

foreach(benchmark IN LISTS memory_benchmarks)
    add_executable(${benchmark} ${benchmark}.cu)
    target_include_directories(${benchmark} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${benchmark} PRIVATE CUDA::cudart)
    if(BUILD_TESTING)
        add_test(NAME memory_${benchmark} COMMAND ${benchmark})
        set_tests_properties(memory_${benchmark} PROPERTIES
            LABELS "memory;benchmark"
            RUN_SERIAL TRUE
            TIMEOUT 120
            PASS_REGULAR_EXPRESSION "\\[SUCCESS\\]"
        )
    endif()
endforeach()

add_custom_target(memory_benchmarks DEPENDS ${memory_benchmarks})
