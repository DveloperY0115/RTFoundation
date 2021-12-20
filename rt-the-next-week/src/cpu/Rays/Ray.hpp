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
    Ray(const Point3& RayOrigin, const Vector3& RayDirection, double CreatedTime = 0.0)
            : Origin(RayOrigin), Direction(RayDirection), Time(CreatedTime)
    {
        // Do nothing
    }

    Vector3 getRayOrigin() const {
        return Origin;
    }

    Vector3 getRayDirection() const {
        return Direction;
    }

    double getCreatedTime() const {
        return Time;
    }

    Vector3 getPointAt(double Depth) const
    {
        return Origin + Depth * Direction;
    }

public:
    Vector3 Origin;
    Vector3 Direction;
    double Time;
};

#endif //RTFOUNDATION_RAY_HPP
