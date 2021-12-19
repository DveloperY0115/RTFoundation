# if OpenMP is available, build binary to support multi-threading/processing
find_package(OpenMP)

# renderer using cpu only
add_executable(rt-the-next-week-cpu main.cpp Cameras/Camera.hpp Colors/Colors.hpp
        Geometry/Hittable.hpp Geometry/HittableList.hpp Materials/Material.hpp Materials/Lambertian.hpp
        Materials/Metal.hpp Materials/Dielectric.hpp Rays/Ray.hpp rtweekend.hpp
        Geometry/Sphere.hpp Math/Vector3.hpp Geometry/MovingSphere.hpp Geometry/AABB.hpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(rt-the-next-week-cpu PUBLIC OpenMP::OpenMP_CXX)
endif()