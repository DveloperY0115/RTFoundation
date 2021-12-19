//
// Created by dveloperY0115 on 12/29/2020.
//

#ifndef RTFOUNDATION_LAMBERTIAN_H
#define RTFOUNDATION_LAMBERTIAN_H

#include "Material.hpp"
#include "../Textures/SolidColor.hpp"

/*
 * Lambertian(diffuse) Material class
 */
class Lambertian : public Material
{
public:
    // Constructor
    Lambertian(const Color& DiffuseColor)
    : Albedo(make_shared<SolidColor>(DiffuseColor))
    {
        // Do nothing.
    }

    Lambertian(shared_ptr<Texture> DiffuseColor)
    : Albedo(DiffuseColor)
    {
        // Do nothing.
    }

    virtual bool scatter(
            const Ray& IncidentRay, const HitRecord& Record, Color& Attenuation, Ray& ScatteredRay
    ) const override
    {
        // randomly set where the ScatteredRay Ray heads
        auto ScatterDirection = Record.HitPointNormal + randomInUnitSphere();

        // catch degenerate scatter getRayDirection - this happens when randomly generated vector is almost against HitPointNormal
        if (ScatterDirection.nearZero())
            ScatterDirection = Record.HitPointNormal;

        /*
         * create actual 'Ray' instance and assign its reference to 'ScatteredRay',
         * also assign Albedo value to 'Attenuation'
         */
        ScatteredRay = Ray(Record.HitPoint, ScatterDirection, IncidentRay.getCreatedTime());
        Attenuation = Albedo->getTexelColor(Record.u, Record.v, Record.HitPoint);
        return true;
    }

public:
    /*
    * Albedo - the factor that determines the portion of incident Ray that the Material reflects
    */
    shared_ptr<Texture> Albedo;
};

#endif //RTFOUNDATION_LAMBERTIAN_H
