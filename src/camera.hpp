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
            point3 lookfrom,
            point3 lookat,
            vector3 vup,
            double vfov, // vertical field-of-view in degrees
            double aspect_ratio
            ) {
            auto theta = degress_to_radians(vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;

            auto w = unit_vector(lookfrom - lookat);
            auto u = unit_vector(cross_product(vup, w));
            auto v = cross_product(w, u);

            auto focal_length = 1.0;

            origin = lookfrom;
            horizontal = viewport_width * u;
            vertical = viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - w;
    }

    ray get_ray(double s, double t) const
    {
        return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;
};

#endif //FIRSTRAYTRACER_CAMERA_HPP
