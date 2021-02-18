//
// Created by dveloperY0115 on 1/28/2021.
//

#ifndef RAY_TRACING_IN_CPP_DIELECTRIC_H
#define RAY_TRACING_IN_CPP_DIELECTRIC_H

#include "rtweekend-gpu.hpp"
#include "material.hpp"

class dielectric: public material {
public:
    __device__ dielectric(double index_of_refraction): ir(index_of_refraction) {}

    __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state
            ) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vector3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vector3 refracted_direction;

        // if refraction cannot happen, reflect instead
        if (cannot_refract || (reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))) {
            refracted_direction = reflect(unit_direction, rec.normal);
        } else {
            refracted_direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = ray(rec.p, refracted_direction, r_in.time());
        return true;
    }

public:
    double ir; // index of refraction

private:
    static __device__ double reflectance(double cosine, double ref_idx) {
        auto r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
    }
};

#endif //RAY_TRACING_IN_CPP_DIELECTRIC_H
