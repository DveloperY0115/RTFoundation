#ifndef CAMERAH
#define CAMERAH

#include "ray.hpp"

class camera {
public:
    __device__ camera() {
        lower_left_corner = vector3(-2.0, -1.0, -1.0);
        horizontal = vector3(4.0, 0.0, 0.0);
        vertical = vector3(0.0, 2.0, 0.0);
        origin = vector3(0.0, 0.0, 0.0);
    }
    __device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); }

    vector3 origin;
    vector3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;
};

#endif