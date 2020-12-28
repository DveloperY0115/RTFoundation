//
// Created by 유승우 on 2020/05/15.
//

#ifndef FIRSTRAYTRACER_CAMERA_HPP
#define FIRSTRAYTRACER_CAMERA_HPP

#include "rtweekend.hpp"

class camera
{
public:
    camera(
            double vfov, // vertical field-of-view in degrees
            double aspect_ratio
            ) {
            auto theta = degress_to_radians(vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;

            auto focal_length = 1.0;

            origin = point3(0, 0, 0);
            horizontal = vector3(viewport_width, 0.0, 0.0);
            vertical = vector3(0.0, viewport_height, 0.0);
            lower_left_corner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focal_length);
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
