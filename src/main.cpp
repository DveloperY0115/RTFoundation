#include <iostream>

#include "camera.hpp"
#include "rtweekend.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"
#include "color.hpp"
#include "material.hpp"
#include "lambertian.hpp"
#include "metal.hpp"
#include "dielectric.hpp"

color ray_color(const ray& r, const hittable& world, int depth)
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

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}


int main()
{
    // configure output image

    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500;
    const int max_depth = 50;

    // set world

    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.6, 1.0, 0.4));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    auto glass_material = make_shared<dielectric>(1.5);

    for (int i = 1; i < 11; i++) {
        if (i % 2 == 1) {
            world.add(make_shared<sphere>(point3(0, 2.5, 0), 0.25 * i, glass_material));
        } else {
            world.add(make_shared<sphere>(point3(0, 2.5, 0), -0.25 * i, glass_material));
        }
    }

    // set camera

    point3 lookfrom(4, 4, 4);
    point3 lookat(0, 2.5, 0);
    vector3 vup(0, 1, 0);
    auto dist_to_focus = 1.0;
    auto aperture = 0.1;

    camera cam = camera(lookfrom, lookat, vup, 55, aspect_ratio, aperture, dist_to_focus);

    // render

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            color pixel_color(0, 0,0 );
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";
}
