//
// Created by 유승우 on 2020/04/24.
//

#include <iostream>
#include "../Include/ray/ray.h"
#include "../Include/vector3/vector3.h"

ray::ray(const vector3& origin, const vector3& direction)
        : orig(origin), dir(direction)
{}

vector3 ray::origin() const
{
    return vector3();
}

vector3 ray::direction() const
{
    return dir;
}

vector3 ray::at(double t) const
{
    return orig + t * dir;
}