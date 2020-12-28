//
// Created by 유승우 on 2020/05/21.
//

#ifndef FIRSTRAYTRACER_MATERIAL_HPP
#define FIRSTRAYTRACER_MATERIAL_HPP


#include "ray.hpp"
#include "hittable.hpp"

/*
 * Generalized, abstract class for different materials
 * The common roles of various material classes are:
 * 1. If the object didn't fully absorbed incident ray, produce a scattered ray
 * 2. (If 1), determine how much the scattered ray is attenuated compared to incident ray
 *
 * In short, the material of the surface tells the ray tracer how rays interact with the surface.
 */
class material
{
public:
    virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
            ) const = 0;
};

/*
 * Metal material class
 */
class metal : public material
{
public:
    // Constructor
    metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f: 1) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
            ) const
    {
        vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

        // since we passed a reference of variable, the scattered ray is determined here.
        // the fuzziness determines the clarity of metal surface. if it's value gets large
        // the metal gets closer to diffuse material
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
        attenuation = albedo;
        return (dot_product(scattered.direction(), rec.normal) > 0);
    }

public:
    color albedo;
    double fuzz;
};

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
    color albedo;
};

#endif //FIRSTRAYTRACER_MATERIAL_HPP
