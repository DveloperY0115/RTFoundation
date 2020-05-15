//
// Created by 유승우 on 2020/04/22.
//

#ifndef FIRSTRAYTRACER_RAY_HPP
#define FIRSTRAYTRACER_RAY_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "vector3.hpp"

class ray{
public:
    ray()
    {
        // Do nothing
    }
    ray(const vector3& origin, const vector3& direction)
            : orig(origin), dir(direction)
    {
        // Do nothing
    }

    vector3 origin() const
    {
        return vector3();
    }

    vector3 direction() const
    {
        return dir;
    }

    vector3 at(double t) const
    {
        return orig + t * dir;
    }

public:
    vector3 orig;
    vector3 dir;
};

#endif //FIRSTRAYTRACER_RAY_HPP
