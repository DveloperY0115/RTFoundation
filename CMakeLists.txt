# CMake version
cmake_minimum_required(VERSION 3.15)

# Declare project
project(FirstRayTracer)

set(CMAKE_CXX_STANDARD 17)
set( CMAKE_C_COMPILER  "gcc" )

add_executable(FirstRayTracer src/main.cpp)