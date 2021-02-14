#ifndef CAMERAH
#define CAMERAH

#include "ray.hpp"

class camera {
public:
    __device__ camera(
            point3 lookfrom,
            point3 lookat,
            vector3 vup,
            double vfov, // vertical field-of-view in degrees
            double aspect_ratio,
            double aperture,
            double focus_dist,
            double _time0 = 0.0,
            double _time1 = 0.0
            ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist * w;

        lens_radius = aperture / 2;
        time0 = _time0;
        time1 = _time1;
    }
    __device__ ray get_ray(float s, float t, curandState* local_rand_state) const {
        vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vector3 offset = u * rd.x() + v * rd.y();

        return ray(
                origin + offset,
                lower_left_corner + s * horizontal + t * vertical - origin - offset,
                curand_uniform(local_rand_state) * (time1 - time0) + time0
        );
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vector3 u, v, w;
    vector3 horizontal;
    vector3 vertical;
    double lens_radius;
    double time0, time1;
};

#endif