//
// Created by dveloperY0115 on 1/8/2021.
//

#ifndef RAY_TRACING_IN_CPP_HITTABLE_H
#define RAY_TRACING_IN_CPP_HITTABLE_H

#include "ray.hpp"

class material;

struct hit_record {
    __device__ inline void set_face_normal(const ray& r, const vector3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }

    float t;
    vector3 p;
    vector3 normal;
    material* mat_ptr;
    bool front_face;
};

class hittable  {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};


#endif //RAY_TRACING_IN_CPP_HITTABLE_H
