//
// Created by 유승우 on 2020/04/22.
//

#ifndef FIRSTRAYTRACER_RAY_H
#define FIRSTRAYTRACER_RAY_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include "vector3/vector3.h"

class ray
{
public:

    vector3 A;
    vector3 B;

    ray();
    ray(const vector3& a, const vector3& b);
    vector3 origin() const;
    vector3 direction() const;
    vector3 point_at_parameter(float t) const;


};

#endif //FIRSTRAYTRACER_RAY_H
