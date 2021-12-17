# renderer using cpu only
add_executable(rt-cpu main.cpp Camera.hpp color.hpp
        hittable.hpp hittable_list.hpp material.hpp lambertian.hpp
        metal.hpp dielectric.hpp ray.hpp rtweekend.hpp
        sphere.hpp Vector3.hpp)