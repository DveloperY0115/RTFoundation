# CMake version
cmake_minimum_required(VERSION 3.15)

# Declare project
project(FirstRayTracer)

set(CMAKE_CXX_STANDARD 17)
set( CMAKE_C_COMPILER  "gcc" )

add_executable(FirstRayTracer src/main.cpp src/camera.hpp src/color.hpp
        src/hittable.hpp src/hittable_list.hpp src/material.hpp src/lambertian.hpp
        src/metal.hpp src/dielectric.hpp src/ray.hpp src/rtweekend.hpp
        src/sphere.hpp src/vector3.hpp)