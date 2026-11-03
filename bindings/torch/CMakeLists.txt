if(NOT TARGET cuda_ops_vector_add)
    message(FATAL_ERROR "当前 Python 示例需要 CUDA_LAB_OPS 包含 vector_add")
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
# 从实际构建解释器读取头文件、库和 ABI，避免寻找另一环境的 LibTorch。
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c [=[
import json
import torch
from torch.utils.cpp_extension import include_paths, library_paths
print(json.dumps({"includes": include_paths(), "libraries": library_paths(),
                  "abi": int(torch.compiled_with_cxx11_abi()),
                  "version": torch.__version__, "cuda": torch.version.cuda}))
]=]
    RESULT_VARIABLE torch_result
    OUTPUT_VARIABLE torch_metadata
    ERROR_VARIABLE torch_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT torch_result EQUAL 0)
    message(FATAL_ERROR "构建解释器无法导入 PyTorch: ${torch_error}")
endif()

add_library(cuda_lab_torch SHARED register.cc vector_add.cc)
target_compile_features(cuda_lab_torch PRIVATE cxx_std_20)
set_target_properties(cuda_lab_torch PROPERTIES PREFIX "" OUTPUT_NAME "_C")

foreach(field includes libraries)
    string(JSON count LENGTH "${torch_metadata}" ${field})
    math(EXPR last "${count} - 1")
    set(torch_${field} "")
    foreach(index RANGE ${last})
        string(JSON value GET "${torch_metadata}" ${field} ${index})
        list(APPEND torch_${field} "${value}")
    endforeach()
endforeach()

string(JSON torch_abi GET "${torch_metadata}" abi)
string(JSON CUDA_LAB_TORCH_VERSION GET "${torch_metadata}" version)
string(JSON CUDA_LAB_TORCH_CUDA GET "${torch_metadata}" cuda)
target_include_directories(cuda_lab_torch SYSTEM PRIVATE ${torch_includes})

if(NOT MSVC)
    target_compile_definitions(cuda_lab_torch PRIVATE _GLIBCXX_USE_CXX11_ABI=${torch_abi})
endif()

foreach(component c10 c10_cuda torch_cpu torch_cuda)
    # 同一 worktree 切换 Python 环境时，重新定位库，避免沿用旧环境的缓存路径。
    unset(cuda_lab_${component} CACHE)
    find_library(cuda_lab_${component} NAMES ${component} PATHS ${torch_libraries} NO_DEFAULT_PATH NO_CACHE REQUIRED)
    target_link_libraries(cuda_lab_torch PRIVATE "${cuda_lab_${component}}")
endforeach()

target_link_libraries(cuda_lab_torch PRIVATE cuda_lab::vector_add)
set_target_properties(cuda_lab_torch PROPERTIES
    BUILD_RPATH "${torch_libraries}"
    INSTALL_RPATH "$ORIGIN/../torch/lib"
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(cuda_lab_torch PRIVATE -Wl,--no-undefined)
endif()

set(CUDA_LAB_GIT_COMMIT "unknown")
set(CUDA_LAB_GIT_DIRTY "null")
find_package(Git QUIET)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE CUDA_LAB_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE git_status)
    if(git_status STREQUAL "")
        set(CUDA_LAB_GIT_DIRTY "false")
    else()
        set(CUDA_LAB_GIT_DIRTY "true")
    endif()
endif()

configure_file(build_info.json.in "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" @ONLY)
install(TARGETS cuda_lab_torch LIBRARY DESTINATION cuda_lab COMPONENT python)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" DESTINATION cuda_lab COMPONENT python)

if(NOT SKBUILD)
    # wheel.packages 管理打包时的 Python 源码，保留 editable 的源码重定向。
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/python/cuda_lab/" DESTINATION cuda_lab COMPONENT python
        FILES_MATCHING PATTERN "*.py")
endif()
