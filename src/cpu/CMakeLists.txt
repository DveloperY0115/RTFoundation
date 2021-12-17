# renderer using cpu only
add_executable(rt-cpu main.cpp Camera.hpp color.hpp
        Hittable.hpp HittableList.hpp Material.hpp Lambertian.hpp
        Metal.hpp Dielectric.hpp Ray.hpp rtweekend.hpp
        Sphere.hpp Vector3.hpp)