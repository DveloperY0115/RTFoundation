//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_CHECKERTEXTURE_HPP
#define RTFOUNDATION_CHECKERTEXTURE_HPP

#include "Texture.hpp"
#include "SolidColor.hpp"

class CheckerTexture : public Texture {
public:
    CheckerTexture();

    CheckerTexture(shared_ptr<Texture> EvenTextureIn, shared_ptr<Texture> OddTextureIn, double Frequency_)
    : EvenTexture(EvenTextureIn), OddTexture(OddTextureIn), Frequency(Frequency_)
    {
        // Do nothing.
    }

    CheckerTexture(Color EvenColor, Color OddColor, double Frequency_)
    : EvenTexture(make_shared<SolidColor>(EvenColor)), OddTexture(make_shared<SolidColor>(OddColor)), Frequency(Frequency_)
    {
        // Do nothing.
    }

    Color getTexelColor(double u, double v, const Point3& SurfacePoint) const override {
        auto Sines = sin(Frequency * SurfacePoint.X()) * sin(Frequency * SurfacePoint.Y()) * sin(Frequency * SurfacePoint.Z());
        if (Sines < 0)
            return OddTexture->getTexelColor(u, v, SurfacePoint);
        else
            return EvenTexture->getTexelColor(u, v, SurfacePoint);
    }

public:
    shared_ptr<Texture> EvenTexture, OddTexture;
    double Frequency;
};

#endif //RTFOUNDATION_CHECKERTEXTURE_HPP
