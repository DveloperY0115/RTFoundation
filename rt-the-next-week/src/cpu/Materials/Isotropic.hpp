//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_ISOTROPIC_HPP
#define RTFOUNDATION_ISOTROPIC_HPP

#include "Material.hpp"
#include "../Textures/SolidColor.hpp"

class Isotropic : public Material {
public:
    Isotropic(Color Color_)
    : Albedo(make_shared<SolidColor>(Color_))
    {
        // Do nothing.
    }

    Isotropic(shared_ptr<Texture> Texture_)
    : Albedo(Texture_)
    {
        // Do nothing.
    }

    bool scatter(const Ray& IncidentRay, const HitRecord& Record, Color& Attenuation, Ray& ScatteredRay) const override {
        ScatteredRay = Ray(Record.HitPoint, randomInUnitSphere(), IncidentRay.getCreatedTime());
        Attenuation = Albedo->getTexelColor(Record.u, Record.v, Record.HitPoint);
        return true;
    }

public:
    shared_ptr<Texture> Albedo;
};

#endif //RTFOUNDATION_ISOTROPIC_HPP
