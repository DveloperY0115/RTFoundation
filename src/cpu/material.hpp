//
// Created by 유승우 on 2020/05/21.
//

#ifndef RTFOUNDATION_MATERIAL_HPP
#define RTFOUNDATION_MATERIAL_HPP


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
            const ray& r_in, const hit_record& rec, Color& attenuation, ray& scattered
            ) const = 0;
};







#endif //RTFOUNDATION_MATERIAL_HPP
