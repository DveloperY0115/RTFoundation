//
// Created by dveloperY0115 on 12/29/2020.
//

#ifndef RTFOUNDATION_METAL_H
#define RTFOUNDATION_METAL_H

#include "Material.hpp"

/*
 * Metal Material class
 */
class Metal : public Material
{
public:
    // Constructor
    Metal(const Color& a, double f) : Albedo(a), Fuzziness(f < 1 ? f : 1) {}

    virtual bool scatter(
            const Ray& IncidentRay, const HitRecord& Record, Color& Attenuation, Ray& ScatteredRay
    ) const override
    {
        // determine the getRayDirection of ReflectedRayDir Ray
        Vector3 ReflectedRayDir = reflect(normalize(IncidentRay.getRayDirection()), Record.HitPointNormal);

        /**
         * generate a Ray object that originates from the point of incidence, and spreads out toward certain getRayDirection
         * the getRayDirection might be randomized according to the fuzziness of this Material
         */
        ScatteredRay = Ray(Record.HitPoint, ReflectedRayDir + Fuzziness * randomInUnitSphere());
        Attenuation = Albedo;
        // if fuzziness is too high, the light may not be ReflectedRayDir off the surface (rather, it seems to be absorbed)
        return (dotProduct(ScatteredRay.getRayDirection(), Record.HitPointNormal) > 0);
    }

public:
    /*
     * Albedo - the factor that determines the portion of incident Ray that the Material reflects
     * Fuzzyness (fuzziness) - the factor of not being clear, Metal with higher fuzziness tends to act similar to diffuse
     */
    Color Albedo;
    double Fuzziness;
};

#endif //RTFOUNDATION_METAL_H
