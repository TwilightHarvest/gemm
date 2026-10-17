cmake_minimum_required(VERSION 3.20)

project(gemm
    VERSION 0.1.0
    LANGUAGES CXX
)

# Must be set before any targets are created: targets copy these at creation.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ninja is single-config; with no build type CMake passes no -O flag at all.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_NATIVE "Tune for the host CPU (-mcpu=native / -march=native)" ON)

# OpenBLAS is the reference for both correctness and speed.
# Homebrew's openblas is keg-only, so point CMake at it.
if(APPLE AND NOT OpenBLAS_DIR)
    execute_process(COMMAND brew --prefix openblas
        OUTPUT_VARIABLE _openblas_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_openblas_prefix)
        list(APPEND CMAKE_PREFIX_PATH ${_openblas_prefix})
    endif()
endif()
find_package(OpenBLAS REQUIRED CONFIG)
if(NOT TARGET OpenBLAS::OpenBLAS)
    add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
    target_include_directories(OpenBLAS::OpenBLAS INTERFACE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(OpenBLAS::OpenBLAS INTERFACE ${OpenBLAS_LIBRARIES})
endif()

# Shared compile settings for every target in the project.
add_library(gemm_options INTERFACE)
target_compile_options(gemm_options INTERFACE -Wall -Wextra -Wpedantic)
if(ENABLE_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        target_compile_options(gemm_options INTERFACE -mcpu=native)
    else()
        target_compile_options(gemm_options INTERFACE -march=native)
    endif()
endif()
if(ENABLE_ASAN)
    target_compile_options(gemm_options INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(gemm_options INTERFACE -fsanitize=address)
endif()

# Kernels live in a library so the bench (and later tests) link the same code.
add_library(gemm_kernels STATIC
    src/naive.cpp
)
target_include_directories(gemm_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gemm_kernels PUBLIC gemm_options)

add_executable(gemm_bench
    src/main.cpp
)
target_link_libraries(gemm_bench PRIVATE gemm_kernels OpenBLAS::OpenBLAS)
