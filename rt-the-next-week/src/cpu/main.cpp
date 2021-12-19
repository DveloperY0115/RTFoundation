#include <array>
#include <iostream>
#include <omp.h>

#include "Cameras/Camera.hpp"
#include "rtweekend.hpp"
#include "Geometry/HittableList.hpp"
#include "Geometry/Sphere.hpp"
#include "Geometry/MovingSphere.hpp"
#include "Colors/Colors.hpp"
#include "Materials/Material.hpp"
#include "Materials/Lambertian.hpp"
#include "Materials/Metal.hpp"
#include "Materials/Dielectric.hpp"

Color computeRayColor(const Ray& r, const Hittable& World, int Depth) {
    HitRecord Record;
    // If we've exceeded the Ray bounce limit, no more light is gathered.
    if (Depth <= 0)
            return Color(0, 0, 0);

    if (World.hit(r, 0.001, Infinity, Record)) {
        Ray ScatteredRay;
        Color Attenuation;
        if (Record.MaterialPtr->scatter(r, Record, Attenuation, ScatteredRay))
            return Attenuation * computeRayColor(ScatteredRay, World, Depth - 1);

        // 'black body' <- the object absorbs all lights
        return Color(0, 0, 0);
    }

    // if the Ray didn'Depth hit any object, set the pixel color as background
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
                    auto center2 = center + Vector3(0, generateRandomDouble(0,.5), 0);
                    world.add(make_shared<MovingSphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));
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


int main() {
    // configure OpenMP
    omp_set_num_threads(8);

    // configure output image
    const auto AspectRatio = 16.0 / 9.0;
    const int ImageWidth = 400;
    const int ImageHeight = static_cast<int>(ImageWidth / AspectRatio);
    const int SamplesPerPixel = 100;
    const int MaxRecursion = 8;

    // set world
    HittableList world = generateRandomScene();

    // set Camera
    Point3 LookFrom(13, 2, 3);
    Point3 LookAt(0, 0, 0);
    Vector3 UpVector(0, 1, 0);
    auto DistanceToFocus = 10.0;
    auto Aperture = 0.1;

    Camera cam = Camera(LookFrom,
                        LookAt,
                        UpVector,
                        20,
                        AspectRatio,
                        Aperture,
                        DistanceToFocus,
                        0.0, 1.0);

    // render
    double start, end;
    std::cerr << "Rendering a " << ImageWidth << "x" << ImageHeight << " image with " << SamplesPerPixel << " samples per pixel\n";
    start = omp_get_wtime();
    std::cout << "P3\n" << ImageWidth << " " << ImageHeight << "\n255\n";

    // initialize image buffer
    int* ImageBuffer = new int[3 * ImageWidth * ImageHeight];

    #pragma omp parallel default(none) firstprivate(ImageHeight, ImageWidth) shared(cam, world, ImageBuffer)
    {
        #pragma omp for // trace rays & compute pixel colors
        for (int j = ImageHeight - 1; j >= 0; --j) {
            for (int i = 0; i < ImageWidth; ++i) {
                Color PixelColor(0, 0, 0);
                for (int s = 0; s < SamplesPerPixel; ++s) {
                    auto u = (i + generateRandomDouble()) / (ImageWidth - 1);
                    auto v = (j + generateRandomDouble()) / (ImageHeight - 1);
                    Ray r = cam.getRay(u, v);
                    PixelColor += computeRayColor(r, world, MaxRecursion);
                }
                writeColor(i, j, PixelColor, SamplesPerPixel, ImageWidth, ImageHeight, ImageBuffer);
            }
        }
    }

    // flush image buffer
    flushBuffer(std::cout, ImageWidth, ImageHeight, ImageBuffer);

    // free allocated memory
    delete[] ImageBuffer;

    end = omp_get_wtime();
    double timer_seconds = ((double)(end - start));

    std::cerr << "\ntook " << timer_seconds << " seconds.\n";
    std::cerr << "\nDone.\n";
}
