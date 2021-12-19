//
// Created by 유승우 on 2021/12/19.
//

#ifndef RTFOUNDATION_TEXTURE_HPP
#define RTFOUNDATION_TEXTURE_HPP

#include "../rtweekend.hpp"

class Texture {
public:
    virtual Color getTexelColor(double u, double v, const Point3& SurfacePoint) const = 0;
};
#endif //RTFOUNDATION_TEXTURE_HPP
