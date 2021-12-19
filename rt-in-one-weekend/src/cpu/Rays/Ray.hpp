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
            : Origin(origin), Direction(direction)
    {
        // Do nothing
    }

    Vector3 getRayOrigin() const
    {
        return Origin;
    }

    Vector3 getRayDirection() const
    {
        return Direction;
    }

    Vector3 getPointAt(double t) const
    {
        return Origin + t * Direction;
    }

public:
    Vector3 Origin;
    Vector3 Direction;
};

#endif //RTFOUNDATION_RAY_HPP
