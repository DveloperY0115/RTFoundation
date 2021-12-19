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

    CheckerTexture(shared_ptr<Texture> EvenTextureIn, shared_ptr<Texture> OddTextureIn)
    : EvenTexture(EvenTextureIn), OddTexture(OddTextureIn)
    {
        // Do nothing.
    }

    CheckerTexture(Color EvenColor, Color OddColor)
    : EvenTexture(make_shared<SolidColor>(EvenColor)), OddTexture(make_shared<SolidColor>(OddColor))
    {
        // Do nothing.
    }

    Color getTexelColor(double u, double v, const Point3& SurfacePoint) const override {
        auto Sines = sin(10 * SurfacePoint.X()) * sin(10* SurfacePoint.Y()) * sin(10* SurfacePoint.Z());
        if (Sines < 0)
            return OddTexture->getTexelColor(u, v, SurfacePoint);
        else
            return EvenTexture->getTexelColor(u, v, SurfacePoint);
    }

public:
    shared_ptr<Texture> EvenTexture, OddTexture;
};

#endif //RTFOUNDATION_CHECKERTEXTURE_HPP
