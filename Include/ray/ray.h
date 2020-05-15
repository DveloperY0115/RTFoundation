//
// Created by 유승우 on 2020/04/22.
//

#ifndef FIRSTRAYTRACER_RAY_H
#define FIRSTRAYTRACER_RAY_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "../Include/vector3/vector3.h"

class ray{
public:
    ray() {}
    ray(const vector3& origin, const vector3& direction);

    vector3 origin() const;
    vector3 direction() const;

    vector3 at(double t) const;

public:
    vector3 orig;
    vector3 dir;
};

#endif //FIRSTRAYTRACER_RAY_H
