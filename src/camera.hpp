//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_CAMERA_HPP
#define FIRSTRAYTRACER_CAMERA_HPP

#include "rtweekend.hpp"

class camera
{
public:
    camera()
    {
         lower_left_corner = point3(-2.0, -1.0, -1.0);      // global coordinate of upper left corner of the viewport
         horizontal = vector3(4.0, 0.0, 0.0);   // horizontal basis within the viewport plane
         vertical = vector3(0.0, 2.0, 0.0);     // vertical basis within the viewport plane
         origin = point3(0.0, 0.0, 0.0);    // global coordinate of camera itself
    }

    ray get_ray(double u, double v) const
    {
        return ray(origin, (lower_left_corner + u * horizontal + v * vertical) - origin);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;
};

#endif //FIRSTRAYTRACER_CAMERA_HPP
