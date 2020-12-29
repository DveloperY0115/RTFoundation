//
// Created by dveloperY0115 on 12/29/2020.
//

#ifndef FIRSTRAYTRACER_LAMBERTIAN_H
#define FIRSTRAYTRACER_LAMBERTIAN_H

#include "material.hpp"

/*
 * Lambertian(diffuse) material class
 */
class lambertian : public material
{
public:
    // Constructor
    lambertian(const color& a) : albedo(a) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
    ) const override
    {
        // randomly set where the scattered ray heads
        auto scatter_direction = rec.normal + random_in_unit_sphere();

        // catch degenerate scatter direction - this happens when randomly generated vector is almost against normal
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        /*
         * create actual 'ray' instance and assign its reference to 'scattered',
         * also assign albedo value to 'attenuation'
         */
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    /*
    * albedo - the factor that determines the portion of incident ray that the material reflects
    */
    color albedo;
};

#endif //FIRSTRAYTRACER_LAMBERTIAN_H
