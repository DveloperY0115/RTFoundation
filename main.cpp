#include <iostream>
#include <fstream>
#include "vector3/vector3.h"
#include "ray/ray.h"

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

vector3 ray_color(const ray& r)
{
    auto t = hit_sphere(vector3(0,0,-1), 0.5, r);
    if (t > 0.0)
    {
        vector3 N = unit_vector(r.at(t) - vector3(0,0,-1));
        return 0.5*vector3(N.getX()+1, N.getY()+1, N.getZ()+1);
    }
    vector3 unit_direction = unit_vector(r.direction());
    t = 0.5*(unit_direction.getY() + 1.0);
    return (1.0-t)*vector3(1.0, 1.0, 1.0) + t*vector3(0.5, 0.7, 1.0);
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
    for (int j = image_height-1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            auto u = double(i) / image_width;
            auto v = double(j) / image_height;
            ray r(origin, lower_left_corner + u*horizontal + v*vertical);
            vector3 color = ray_color(r);
            color.write_color(std::cout);
        }
    }

    std::cerr << "\nDone.\n";
}
