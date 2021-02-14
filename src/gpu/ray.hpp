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
    __device__ ray() {
        // Do nothing
    }
    __device__ ray(const vector3& origin, const vector3& direction, double time = 0.0)
            : orig(origin), dir(direction), tm(time) {
        // Do nothing
    }

    __device__ vector3 origin() const {
        return orig;
    }

    __device__ vector3 direction() const {
        return dir;
    }

    __device__ vector3 at(float t) const {
        return orig + t * dir;
    }

    __device__ double time() const {
        return tm;
    }

public:
    vector3 orig;
    vector3 dir;
    double tm;
};

#endif //FIRSTRAYTRACER_RAY_HPP
