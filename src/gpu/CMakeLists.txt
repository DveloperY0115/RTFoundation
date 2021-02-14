# renderer using gpu acceleration
add_executable(rt-cuda main.cu rtweekend-gpu.hpp vector3.hpp ray.hpp camera.hpp
        hittable.hpp sphere.hpp hittable_list.hpp material.hpp lambertian.hpp
        metal.hpp dielectric.hpp)

set_target_properties(
        rt-cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)