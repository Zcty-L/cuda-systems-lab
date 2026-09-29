add_executable(device_query device_query.cu)
target_include_directories(device_query PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(device_query PRIVATE CUDA::cudart)
