//
// Created by dveloperY0115 on 1/28/2021.
//

#ifndef RAY_TRACING_IN_CPP_METAL_H
#define RAY_TRACING_IN_CPP_METAL_H

#include "material.hpp"

class metal : public material {
public:
    __device__ metal(const vector3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation,
                                    ray& scattered, curandState *local_rand_state) const  {
        vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vector3 albedo;
    float fuzz;
};

#endif //RAY_TRACING_IN_CPP_METAL_H
