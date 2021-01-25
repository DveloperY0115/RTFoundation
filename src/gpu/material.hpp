//
// Created by dveloperY0115 on 1/25/2021.
//

#ifndef RAY_TRACING_IN_CPP_MATERIAL_H
#define RAY_TRACING_IN_CPP_MATERIAL_H

struct hit_record;

#include "ray.hpp"
#include "hittable.hpp"

#define RANDVEC3 vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
vector3 p;
do {
p = 2.0f * RANDVEC3 - vector3(1, 1, 1);
} while (p.squared_length() >= 1.0f);
return p;
}

__device__ vector3 reflect(const vector3& v, const vector3& n) {
    return v - 2.0f * dot(v, n) * n;
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const vector3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const  {
        vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target-rec.p);
        attenuation = albedo;
        return true;
    }

    vector3 albedo;
};

class metal : public material {
public:
    __device__ metal(const vector3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const  {
        vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vector3 albedo;
    float fuzz;
};

#endif //RAY_TRACING_IN_CPP_MATERIAL_H
