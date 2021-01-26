//
// Created by dveloperY0115 on 1/9/2021.
//

#include <iostream>
#include "vector3.hpp"

__device__ color ray_color(const ray& r, const hittable& world, int depth)
{
    hit_record rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    if (world.hit(r, 0.001, infinity, rec))
    {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);

        // 'black body' <- the object absorbs all lights
        return color(0, 0,0);


        point3 target = rec.p + random_in_hemisphere(rec.normal);
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
    }

    // if the ray didn't hit any object, set the pixel color as background
    // (in this case, background color is set to be a color gradient of sky blue)
    vector3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.getY() + 1.0);
    return (1.0-t)*vector3(1.0, 1.0, 1.0) + t*vector3(0.5, 0.7, 1.0);
}

int main() {
    vector3 vec1 = vector3();
}