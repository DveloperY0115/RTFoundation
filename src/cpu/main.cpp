#include <iostream>
#include <time.h>
#include "Camera.hpp"
#include "rtweekend.hpp"
#include "HittableList.hpp"
#include "Sphere.hpp"
#include "color.hpp"
#include "Material.hpp"
#include "Lambertian.hpp"
#include "Metal.hpp"
#include "Dielectric.hpp"

Color computeRayColor(const Ray& r, const Hittable& world, int depth)
{
    HitRecord Record;
    // If we've exceeded the Ray bounce limit, no more light is gathered.
    if (depth <= 0)
            return Color(0, 0, 0);

    if (world.Hit(r, 0.001, Infinity, Record))
    {
        Ray scatteredRay;
        Color Attenuation;
        if (Record.MaterialPtr->Scatter(r, Record, Attenuation, scatteredRay))
            return Attenuation * computeRayColor(scatteredRay, world, depth - 1);

        // 'black body' <- the object absorbs all lights
        return Color(0, 0, 0);
    }

    // if the Ray didn'Depth Hit any object, set the pixel color as background
    Vector3 UnitDirection = normalize(r.getRayDirection());
    auto t = 0.5*(UnitDirection.Y() + 1.0);
    return (1.0 - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
}

HittableList generateRandomScene() {
    HittableList world;

    auto GroundMaterial = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, GroundMaterial));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto MaterialSelector = generateRandomDouble();
            Point3 center(a + 0.9 * generateRandomDouble(), 0.2, b + 0.9 * generateRandomDouble());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;

                if (MaterialSelector < 0.8) {
                    // diffuse
                    auto albedo = Color::random() * Color::random();
                    sphere_material = make_shared<Lambertian>(albedo);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else if (MaterialSelector < 0.95) {
                    // Metal
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = generateRandomDouble(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<Dielectric>(1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<Dielectric>(1.5);
    world.add(make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

    return world;
}


int main()
{
    // configure output image

    const auto AspectRatio = 16.0 / 9.0;
    const int ImageWidth = 400;
    const int ImageHeight = static_cast<int>(ImageWidth / AspectRatio);
    const int SamplesPerPixel = 500;
    const int MaxRecursion = 50;

    // set world

    HittableList world;

    world = generateRandomScene();

    // set Camera
    Point3 LookFrom(13, 2, 3);
    Point3 LookAt(0, 0, 0);
    Vector3 UpVector(0, 1, 0);
    auto DistanceToFocus = 10.0;
    auto Aperture = 0.1;

    Camera cam = Camera(LookFrom, LookAt, UpVector, 20, AspectRatio, Aperture, DistanceToFocus);

    // render
    clock_t start, end;
    std::cerr << "Rendering a " << ImageWidth << "x" << ImageHeight << " image with " << 50 << " samples per pixel\n";
    start = clock();
    std::cout << "P3\n" << ImageWidth << " " << ImageHeight << "\n255\n";

    for (int j = ImageHeight - 1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < ImageWidth; ++i)
        {
            Color PixelColor(0, 0, 0 );
            for (int s = 0; s < SamplesPerPixel; ++s)
            {
                auto u = (i + generateRandomDouble()) / (ImageWidth - 1);
                auto v = (j + generateRandomDouble()) / (ImageHeight - 1);
                Ray r = cam.getRay(u, v);
                PixelColor += computeRayColor(r, world, MaxRecursion);
            }
            writeColor(std::cout, PixelColor, SamplesPerPixel);
        }
    }
    end = clock();
    double timer_seconds = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cerr << "\ntook " << timer_seconds << " seconds.\n";
    std::cerr << "\nDone.\n";
}
