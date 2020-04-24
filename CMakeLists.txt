# CMake version
cmake_minimum_required(VERSION 3.15)

# Declare project
project(FirstRayTracer)

set(CMAKE_CXX_STANDARD 17)
set( CMAKE_C_COMPILER  "gcc" )

INCLUDE_DIRECTORIES( include FirstRayTracer/Include )
add_executable(FirstRayTracer main.cpp SRC/vector3/vector3.cpp)
