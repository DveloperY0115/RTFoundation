//
// Created by 유승우 on 2021/12/19.
//

#ifndef RTFOUNDATION_SOLIDCOLOR_HPP
#define RTFOUNDATION_SOLIDCOLOR_HPP

#include "Texture.hpp"

class SolidColor : public Texture {
public:
    SolidColor();
    explicit SolidColor(Color Color)
    : ColorValue(Color)
    {
        // Do nothing.
    }

    // wait... shouldn't we check whether intensities lie in [0, 255]?
    SolidColor(double R, double G, double B)
    : SolidColor(Color(R, G, B))
    {
        // Do nothing.
    }

    Color getTexelColor(double u, double v, const Point3& SurfacePoint) const override {
        return ColorValue;
    }

private:
    Color ColorValue;
};

#endif //RTFOUNDATION_SOLIDCOLOR_HPP
