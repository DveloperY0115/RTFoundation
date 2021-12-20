#include <array>
#include <iostream>
#include <omp.h>

#include "Cameras/Camera.hpp"
#include "rtweekend.hpp"
#include "Geometry/HittableList.hpp"
#include "Geometry/Sphere.hpp"
#include "Geometry/MovingSphere.hpp"
#include "Geometry/BVH.hpp"
#include "Colors/Colors.hpp"
#include "Materials/Material.hpp"
#include "Materials/Lambertian.hpp"
#include "Materials/Metal.hpp"
#include "Materials/Dielectric.hpp"
#include "Textures/SolidColor.hpp"
#include "Textures/CheckerTexture.hpp"
#include "Textures/ImageTexture.hpp"

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
    HittableList World;

    auto GroundMaterial = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    World.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, GroundMaterial));

    // auto checker = make_shared<CheckerTexture>(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));
    // World.add(make_shared<Sphere>(Point3(0,-1000,0), 1000, make_shared<Lambertian>(checker)));

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
                    World.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else if (MaterialSelector < 0.95) {
                    // Metal
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = generateRandomDouble(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    World.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<Dielectric>(1.5);
                    World.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<Dielectric>(1.5);
    World.add(make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    World.add(make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    World.add(make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

    return World;
}

HittableList generateRandomSceneMotionBlur() {
    HittableList world;

    // auto GroundMaterial = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    // world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, GroundMaterial));

    auto checker = make_shared<CheckerTexture>(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));
    world.add(make_shared<Sphere>(Point3(0,-1000,0), 1000, make_shared<Lambertian>(checker)));

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

HittableList generateTwoSpheres() {
    HittableList Objects;

    auto Checker = make_shared<CheckerTexture>(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));

    Objects.add(make_shared<Sphere>(Point3(0,-10, 0), 10, make_shared<Lambertian>(Checker)));
    Objects.add(make_shared<Sphere>(Point3(0, 10, 0), 10, make_shared<Lambertian>(Checker)));

    return Objects;
}

HittableList generateEarth() {
    auto EarthTexture = make_shared<ImageTexture>("../../../../data/earthmap.jpeg");
    auto EarthSurface = make_shared<Lambertian>(EarthTexture);
    auto Globe = make_shared<Sphere>(Point3(0,0,0), 2,EarthSurface);

    return HittableList(Globe);
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

    // set Camera
    Point3 LookFrom;
    Point3 LookAt;
    Vector3 UpVector;
    double VerticalFOV = 40.0;
    double Aperture = 0.0;

    // set world
    HittableList World;
    int SceneSelector = 3;

    switch (SceneSelector) {
        case 0:
            World = generateRandomScene();
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            Aperture = 0.1;
            break;

        case 1:
            World = generateRandomSceneMotionBlur();
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            Aperture = 0.1;
            break;

        case 2:
            World = generateTwoSpheres();
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            break;

        case 3:
            World = generateEarth();
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            break;
    }

    UpVector = Vector3(0, 1, 0);
    auto DistanceToFocus = 10.0;

    Camera cam = Camera(LookFrom,
                        LookAt,
                        UpVector,
                        VerticalFOV,
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

    #pragma omp parallel default(none) firstprivate(ImageHeight, ImageWidth) shared(cam, World, ImageBuffer)
    {
        #pragma omp for // trace rays & compute pixel colors
        for (int j = ImageHeight - 1; j >= 0; --j) {
            for (int i = 0; i < ImageWidth; ++i) {
                Color PixelColor(0, 0, 0);
                for (int s = 0; s < SamplesPerPixel; ++s) {
                    auto u = (i + generateRandomDouble()) / (ImageWidth - 1);
                    auto v = (j + generateRandomDouble()) / (ImageHeight - 1);
                    Ray r = cam.getRay(u, v);
                    PixelColor += computeRayColor(r, World, MaxRecursion);
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
