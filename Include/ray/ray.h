//
// Created by 유승우 on 2020/04/22.
//

#ifndef FIRSTRAYTRACER_RAY_H
#define FIRSTRAYTRACER_RAY_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "vector3/vector3.h"

class ray{
public:
    ray() {}
    ray(const vector3& origin, const vector3& direction)
            : orig(origin), dir(direction)
    {}

    vector3 origin() const    { return orig; }
    vector3 direction() const { return dir; }

    vector3 at(double t) const {
        return orig + t*dir;
    }

public:
    vector3 orig;
    vector3 dir;
};

#endif //FIRSTRAYTRACER_RAY_H
