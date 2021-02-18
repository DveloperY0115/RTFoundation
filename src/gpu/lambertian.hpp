//
// Created by dveloperY0115 on 1/28/2021.
//

#ifndef RAY_TRACING_IN_CPP_LAMBERTIAN_H
#define RAY_TRACING_IN_CPP_LAMBERTIAN_H

#include "material.hpp"

class lambertian : public material {
public:
    __device__ lambertian(const vector3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const  {
        vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target-rec.p, r_in.time());
        attenuation = albedo;
        return true;
    }

    vector3 albedo;
};

#endif //RAY_TRACING_IN_CPP_LAMBERTIAN_H
