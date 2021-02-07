cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ray-tracing-in-cpp CXX)

add_subdirectory(src/cpu)
# add_subdirectory(tests)

# if Windows, enable both C++, CUDA build toolchains
if (MSVC)
    project(ray-tracing-in-cpp CXX CUDA)
    if (NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED  True)
    endif()

    add_subdirectory(src/gpu)
endif()