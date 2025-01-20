cmake_minimum_required(VERSION 3.15)
project(SPMP-GEMM CXX)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Use Clang
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

# Set Build Type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set Compiler Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add LLVM Flags
execute_process(
    COMMAND llvm-config --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")

# Core Library Sources
set(CORE_SOURCES
    src/core/weight_memory.cpp
    src/core/processing_element.cpp
    src/core/simd_engine.cpp
)

# Create Python Module
pybind11_add_module(spmp_gemm 
    src/python/bindings.cpp
    ${CORE_SOURCES}
)

target_include_directories(spmp_gemm PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Create Executable Target (optional)
add_executable(SPMP-GEMM ${CORE_SOURCES} src/main.cpp)
target_include_directories(SPMP-GEMM PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
