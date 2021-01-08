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
            double aspect_ratio,
            double aperture,
            double focus_dist
            ) {
            auto theta = degress_to_radians(vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross_product(vup, w));
            v = cross_product(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist * w;

            lens_radius = aperture / 2;
    }

    ray get_ray(double s, double t) const
    {
        vector3 rd = lens_radius * random_in_unit_disk();
        vector3 offset = u * rd.getX() + v * rd.getY();

        return ray(
                origin + offset,
                lower_left_corner + s * horizontal + t * vertical - origin - offset
                );
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vector3 u, v, w;
    vector3 horizontal;
    vector3 vertical;
    double lens_radius;
};

#endif //FIRSTRAYTRACER_CAMERA_HPP
