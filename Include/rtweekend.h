//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_RTWEEKEND_H
#define FIRSTRAYTRACER_RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degress_to_radians(double degrees)
{
    return degrees * (pi / 180);
}

// Common Headers

#include "ray/ray.h"
#include "vector3/vector3.h"

#endif //FIRSTRAYTRACER_RTWEEKEND_H
