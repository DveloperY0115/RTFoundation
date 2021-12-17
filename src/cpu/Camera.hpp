//
// Created by 유승우 on 2020/05/15.
//

#ifndef RTFOUNDATION_CAMERA_HPP
#define RTFOUNDATION_CAMERA_HPP

#include "rtweekend.hpp"

class Camera
{
public:
    Camera(
            Point3 lookfrom,
            Point3 lookat,
            Vector3 vup,
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
        Vector3 rd = lens_radius * random_in_unit_disk();
        Vector3 offset = u * rd.X() + v * rd.Y();

        return ray(
                origin + offset,
                lower_left_corner + s * horizontal + t * vertical - origin - offset
                );
    }

private:
    Point3 origin;
    Point3 lower_left_corner;
    Vector3 u, v, w;
    Vector3 horizontal;
    Vector3 vertical;
    double lens_radius;
};

#endif //RTFOUNDATION_CAMERA_HPP
