cmake_minimum_required(VERSION 3.20)

project(gemm
    VERSION 0.1.0
    LANGUAGES CXX
)

add_executable(${PROJECT_NAME}
    src/main.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    target_compile_options(${PROJECT_NAME} PRIVATE
        -fsanitize=address
        -fno-omit-frame-pointer
    )

    target_link_options(${PROJECT_NAME} PRIVATE
        -fsanitize=address
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
