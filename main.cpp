#include "../Include/rtweekend.h"

#include "../Include/hittable/hittable_list.h"
#include "../Include/geometry/sphere.h"
#include "../Include/color.h"

#include <iostream>

color ray_color(const ray& r, const hittable& world)
{
    hit_record rec;
    if (world.hit(r, 0, infinity, rec))
    {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }
    vector3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.getY() + 1.0);
    return (1.0-t)*vector3(1.0, 1.0, 1.0) + t*vector3(0.5, 0.7, 1.0);
}

using namespace std;

double hit_sphere(const vector3& center, double radius, const ray& r)
{
    vector3 oc = r.origin() - center;
    auto a = dot_product(r.direction(), r.direction());
    auto b = 2.0 * dot_product(oc, r.direction());
    auto c = dot_product(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    if (discriminant < 0)
    {
        return -1.0;
    }

    else
    {
        return (-b - sqrt(discriminant) ) / (2.0*a);
    }
}

int main()
{
    const int image_width = 200;
    const int image_height = 100;

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    vector3 lower_left_corner(-2.0, -1.0, -1.0);
    vector3 horizontal(4.0, 0.0, 0.0);
    vector3 vertical(0.0, 2.0, 0.0);
    vector3 origin(0.0, 0.0, 0.0);

    hittable_list world;
    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));
    for (int j = image_height-1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            auto u = double(i) / image_width;
            auto v = double(j) / image_height;
            ray r(origin, lower_left_corner + u*horizontal + v*vertical);

            color pixel_color = ray_color(r, world);

            write_color(std::cout, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";
}
