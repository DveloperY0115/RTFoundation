#include <iostream>
#include <time.h>
#include "Camera.hpp"
#include "rtweekend.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"
#include "color.hpp"
#include "material.hpp"
#include "lambertian.hpp"
#include "metal.hpp"
#include "dielectric.hpp"

Color ray_color(const ray& r, const hittable& world, int depth)
{
    hit_record rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
            return Color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec))
    {
        ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);

        // 'black body' <- the object absorbs all lights
        return Color(0, 0, 0);

        Point3 target = rec.p + random_in_hemisphere(rec.normal);
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
    }

    // if the ray didn't hit any object, set the pixel Color as background
    // (in this case, background Color is set to be a Color gradient of sky blue)
    Vector3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.Y() + 1.0);
    return (1.0-t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(Point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            Point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = Color::random() * Color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = Color::random(0.5, 1);
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
    world.add(make_shared<sphere>(Point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(Color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(Point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(Point3(4, 1, 0), 1.0, material3));

    return world;
}


int main()
{
    // configure output image

    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500;
    const int max_depth = 50;

    // set world

    hittable_list world;

    // auto ground_material = make_shared<lambertian>(Color(0.6, 1.0, 0.4));
    // auto diffuse_material = make_shared<lambertian>(Color(0.5, 0.5, 0.5));
    // auto glass_material = make_shared<dielectric>(0.5);
    // world.add(make_shared<sphere>(Point3(0, -100, -1), 100, diffuse_material));

    /*
    for (int i = 1; i < 11; i++) {
        if (i % 2 == 1) {
            world.add(make_shared<sphere>(Point3(0, 2.5, 0), 0.25 * i, glass_material));
        } else {
            world.add(make_shared<sphere>(Point3(0, 2.5, 0), -0.25 * i, glass_material));
        }
    }
    */

    // world.add(make_shared<sphere>(Point3(0, 1, -1), 1, diffuse_material));

    world = random_scene();

    // set Camera
    Point3 lookfrom(13,2,3);
    Point3 lookat(0,0,0);
    Vector3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    Camera cam = Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // render
    clock_t start, end;
    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << 50 << " samples per pixel\n";
    start = clock();
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            Color pixel_color(0, 0, 0 );
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
    end = clock();
    double timer_seconds = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cerr << "\ntook " << timer_seconds << " seconds.\n";
    std::cerr << "\nDone.\n";
}
