//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_DIFFUSELIGHT_HPP
#define RTFOUNDATION_DIFFUSELIGHT_HPP

#include "Material.hpp"
#include "../Textures/SolidColor.hpp"

class DiffuseLight : public Material {
public:
    DiffuseLight(shared_ptr<Texture> LightColorTexture)
    : EmittedColorTexture(LightColorTexture)
    {
        // Do nothing.
    }
    DiffuseLight(Color LightColor)
    : EmittedColorTexture(make_shared<SolidColor>(LightColor))
    {

    }

    bool scatter(const Ray& IncidentRay, const HitRecord& Record, Color& Attenuation, Ray& ScatteredRay) const override {
        return false;
    }

    Color emit(double u, double v, const Point3& SurfacePoint) const override {
        return EmittedColorTexture->getTexelColor(u, v, SurfacePoint);
    }

public:
    shared_ptr<Texture> EmittedColorTexture;
};

#endif //RTFOUNDATION_DIFFUSELIGHT_HPP
