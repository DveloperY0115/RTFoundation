//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_CAMERA_HPP
#define FIRSTRAYTRACER_CAMERA_HPP

#include "rtweekend.hpp"

class camera
{
private:
    point3 origin;
    point3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;

public:
    camera()
    {
         lower_left_corner = point3(-2.0, -1.0, -1.0);
         horizontal = vector3(4.0, 0.0, 0.0);
         vertical = vector3(0.0, 2.0, 0.0);
         origin = point3(0.0, 0.0, 0.0);
    }

    ray get_ray(double u, double v) const
    {
        return ray(origin, (lower_left_corner + u * horizontal + v * vertical) - origin);
    }

};

#endif //FIRSTRAYTRACER_CAMERA_HPP
