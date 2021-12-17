//
// Created by dveloperY0115 on 12/29/2020.
//

#ifndef FIRSTRAYTRACER_METAL_H
#define FIRSTRAYTRACER_METAL_H

#include "material.hpp"

/*
 * Metal material class
 */
class metal : public material
{
public:
    // Constructor
    metal(const Color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, Color& attenuation, ray& scattered
    ) const override
    {
        // determine the direction of reflected ray
        Vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

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
    Color albedo;
    double fuzz;
};

#endif //FIRSTRAYTRACER_METAL_H
