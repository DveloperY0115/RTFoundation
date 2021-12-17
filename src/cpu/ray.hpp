//
// Created by 유승우 on 2020/04/22.
//

#ifndef FIRSTRAYTRACER_RAY_HPP
#define FIRSTRAYTRACER_RAY_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Vector3.hpp"

class ray{
public:
    ray()
    {
        // Do nothing
    }
    ray(const Point3& origin, const Vector3& direction)
            : orig(origin), dir(direction)
    {
        // Do nothing
    }

    Vector3 origin() const
    {
        return orig;
    }

    Vector3 direction() const
    {
        return dir;
    }

    Vector3 at(double t) const
    {
        return orig + t * dir;
    }

public:
    Vector3 orig;
    Vector3 dir;
};

#endif //FIRSTRAYTRACER_RAY_HPP
