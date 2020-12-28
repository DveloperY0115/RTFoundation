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
            ) const override
    {
        // determine the direction of reflected ray
        vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

        /**
         * generate a ray object that originates from the point of incidence, and spreads out toward certain direction
         * the direction might be randomized according to the fuzziness of this material
         */
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        // if fuzziness is too high, the light may not be reflected off the surface (rather, it seems to be absorbed)
        return (dot_product(scattered.direction(), rec.normal) > 0);
    }

public:
    /*
     * albedo - the factor that determines the portion of incident ray that the material reflects
     * fuzz (fuzziness) - the factor of not being clear, metal with higher fuzziness tends to act similar to diffuse
     */
    color albedo;
    double fuzz;
};

/*
 * Dielectric material class
 */
class dielectric: public material {
public:
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
            ) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vector3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot_product(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vector3 refracted_direction;

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

#endif //FIRSTRAYTRACER_MATERIAL_HPP
