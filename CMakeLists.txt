cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Enable both C++, CUDA build toolchains
project(ray-tracing-in-cpp CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)

# renderer using cpu only
add_executable(rt-cpu src/cpu/main.cpp src/cpu/camera.hpp src/cpu/color.hpp
        src/cpu/hittable.hpp src/cpu/hittable_list.hpp src/cpu/material.hpp src/cpu/lambertian.hpp
        src/cpu/metal.hpp src/cpu/dielectric.hpp src/cpu/ray.hpp src/cpu/rtweekend.hpp
        src/cpu/sphere.hpp src/cpu/vector3.hpp)

# renderer using gpu acceleration
add_executable(rt-cuda src/gpu/main.cu)

set_target_properties(
        rt-cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)