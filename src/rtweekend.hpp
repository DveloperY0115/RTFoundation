//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_RTWEEKEND_HPP
#define FIRSTRAYTRACER_RTWEEKEND_HPP

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double clamp(double x, double min, double max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline double modern_random_double()
{
    // Newly added random number generation engine.
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    static std::function<double()> rand_generator =
            std::bind(distribution, generator);

    return rand_generator();
}

inline double random_double()
{
    // Returns a random real number in [0, 1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max)
{
    // Returns a random real number in [min, max)
    return min + (max-min)*random_double();
}

inline double degress_to_radians(double degrees)
{
    return degrees * (pi / 180);
}

// Common Headers
#include "ray.hpp"
#include "vector3.hpp"

#endif //FIRSTRAYTRACER_RTWEEKEND_HPP
