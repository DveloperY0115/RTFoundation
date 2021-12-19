//
// Created by dveloperY0115 on 1/28/2021.
//

#ifndef RAY_TRACING_IN_CPP_RTWEEKEND_GPU_H
#define RAY_TRACING_IN_CPP_RTWEEKEND_GPU_H

//!
//! Commonly used headers
#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

__device__ const double pi = 3.1415926535897932385;

//!
//!\brief Generate and return a double type random number
//!\param local_random_state pointer to a CUDA random state instance
//! \return a random double number between 0 and 1
__device__ double random_double(curandState* local_random_state) {
    return (double)curand_uniform(local_random_state);
}

//!
//!\brief Converts the angle represented in degrees to radian
//!\param degrees an angle in degrees
//!\return same angle but in radian
__device__ inline double degrees_to_radians(double degrees)
{
    return degrees * (pi / 180);
}
#endif //RAY_TRACING_IN_CPP_RTWEEKEND_GPU_H
