//
// Created by 유승우 on 2020/05/21.
//

#ifndef RTFOUNDATION_MATERIAL_HPP
#define RTFOUNDATION_MATERIAL_HPP


#include "../Rays/Ray.hpp"
#include "../Geometry/Hittable.hpp"

/*
 * Generalized, abstract class for different materials
 * The common roles of various Material classes are:
 * 1. If the object didn'Depth fully absorbed incident Ray, produce a scattered Ray
 * 2. (If 1), determine how much the scattered Ray is attenuated compared to incident Ray
 *
 * In short, the Material of the surface tells the Ray tracer how rays interact with the surface.
 */
class Material
{
public:
    virtual bool scatter(
            const Ray& IncidentRay, const HitRecord& Record, Color& Attenuation, Ray& ScatteredRay
            ) const = 0;
};







#endif //RTFOUNDATION_MATERIAL_HPP
