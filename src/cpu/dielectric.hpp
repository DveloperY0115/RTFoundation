//
// Created by dveloperY0115 on 12/29/2020.
//

#ifndef FIRSTRAYTRACER_DIELECTRIC_H
#define FIRSTRAYTRACER_DIELECTRIC_H

#include "material.hpp"

/*
 * Dielectric material class
 */
class dielectric: public material {
public:
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, Color& attenuation, ray& scattered
    ) const override {
        attenuation = Color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;   // if incident ray is from the inside of dielectric, (ratio of IOR) = (index of dielectric) / 1.0 (air)

        Vector3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot_product(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vector3 refracted_direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
            refracted_direction = reflect(unit_direction, rec.normal);
        } else {
            refracted_direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = ray(rec.p, refracted_direction);
        return true;
    }

public:
    double ir;  // index of refraction

private:
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif //FIRSTRAYTRACER_DIELECTRIC_H
