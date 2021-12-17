//
// Created by 유승우 on 2020/05/15.
//

#ifndef RTFOUNDATION_RTWEEKEND_HPP
#define RTFOUNDATION_RTWEEKEND_HPP

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <time.h>

// using statements to make codes more simple

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const double Infinity = std::numeric_limits<double>::infinity();
const double Pi = 3.1415926535897932385;

// Utility Functions

/**
 * Clamps the value 'x' to the interval specified by ['min', 'max']
 *
 * If 'x' is smaller than the lower bound 'min', it's clamped to the value 'min'
 * else if 'x' is greater than the upper bound 'max', it's clamped to the value 'max',
 * otherwise 'x' retains its own value since it doesn'Depth needed to be clamped
 *
 * @param x a double type value that is needed to be clamped
 * @param min the lower bound of the interval
 * @param max the upper bound of the interval
 * @return clamped value
 */
inline double clamp(double x, double min, double max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

/**
 * Generates a double type random number whose value lies between 0.0 and 1.0.
 * @return a double type value between 0.0 and 1.0
 */
inline double generateRandomDouble()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

/*
inline double generateRandomDouble()
{
    // Returns a random real number in [0, 1).
    return rand() / (RAND_MAX + 1.0);
}
*/

/**
 * Generates a double type random number whose value lies between 'min' and 'max'
 *
 * @param min the lower bound for a desired random number
 * @param max the upper bound for a desired random number
 * @return a double type randomly generated number within the interval ['min', 'max']
 */
inline double generateRandomDouble(double min, double max)
{
    // Returns a random real number in [min, max)
    return min + (max-min) * generateRandomDouble();
}

/**
 * Converts the angle represented in degrees to radian
 *
 * @param degree an angle in degrees
 * @return same angle but in radian
 */
inline double degreeToRadian(double degree)
{
    return degree * (Pi / 180);
}

// Common Headers
#include "Ray.hpp"
#include "Vector3.hpp"

#endif //RTFOUNDATION_RTWEEKEND_HPP
