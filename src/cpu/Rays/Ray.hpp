//
// Created by 유승우 on 2020/04/22.
//

#ifndef RTFOUNDATION_RAY_HPP
#define RTFOUNDATION_RAY_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "../Math/Vector3.hpp"

class Ray{
public:
    Ray()
    {
        // Do nothing
    }
    Ray(const Point3& origin, const Vector3& direction)
            : orig(origin), dir(direction)
    {
        // Do nothing
    }

    Vector3 getRayOrigin() const
    {
        return orig;
    }

    Vector3 getRayDirection() const
    {
        return dir;
    }

    Vector3 getPointAt(double t) const
    {
        return orig + t * dir;
    }

public:
    Vector3 orig;
    Vector3 dir;
};

#endif //RTFOUNDATION_RAY_HPP
