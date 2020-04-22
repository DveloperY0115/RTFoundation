//
// Created by 유승우 on 2020/04/22.
//

#include "ray/ray.h"

    ray::ray() { }
    ray::ray(const vector3& a, const vector3& b) { A = a; B = b; }

    vector3 ray::origin() const { return A; }
    vector3 ray::direction() const { return B; }
    vector3 ray::point_at_parameter(float t) const { return A + t * B; }

