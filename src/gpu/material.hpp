//
// Created by dveloperY0115 on 1/25/2021.
//

#ifndef RAY_TRACING_IN_CPP_MATERIAL_H
#define RAY_TRACING_IN_CPP_MATERIAL_H

struct hit_record;

#include "ray.hpp"
#include "hittable.hpp"

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation,
                                    ray& scattered, curandState *local_rand_state) const = 0;
};

#endif //RAY_TRACING_IN_CPP_MATERIAL_H
