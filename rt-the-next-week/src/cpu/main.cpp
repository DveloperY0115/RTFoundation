#include <array>
#include <iostream>
#include <omp.h>

#include "Cameras/Camera.hpp"
#include "rtweekend.hpp"
#include "Geometry/HittableList.hpp"
#include "Geometry/Sphere.hpp"
#include "Geometry/AABBRectangle.hpp"
#include "Geometry/MovingSphere.hpp"
#include "Geometry/BVH.hpp"
#include "Colors/Colors.hpp"
#include "Materials/Material.hpp"
#include "Materials/Lambertian.hpp"
#include "Materials/Metal.hpp"
#include "Materials/Dielectric.hpp"
#include "Materials/DiffuseLight.hpp"
#include "Textures/SolidColor.hpp"
#include "Textures/CheckerTexture.hpp"
#include "Textures/ImageTexture.hpp"

Color computeRayColor(const Ray& r, const Color& BackgroundColor, const Hittable& World, int RecursionDepth) {
    HitRecord Record;
    // If we've exceeded the Ray bounce limit, no more light is gathered.
    if (RecursionDepth <= 0)
        return Color(0, 0, 0);

    // If the ray did not hit anything, set its color as the background color.
    if (!World.hit(r, 0.001, Infinity, Record))
        return BackgroundColor;

    Ray ScatteredRay;
    Color Attenuation;
    Color EmittedColor = Record.MaterialPtr->emit(Record.u, Record.v, Record.HitPoint);

    // If the object hit by the ray does not scatter any secondary ray, it's a pure light source.
    if (!Record.MaterialPtr->scatter(r, Record, Attenuation, ScatteredRay))
        return EmittedColor;

    return EmittedColor + Attenuation * computeRayColor(ScatteredRay, BackgroundColor, World, RecursionDepth - 1);
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

HittableList generateSimpleLight() {
    HittableList objects;

    auto earthTexture = make_shared<ImageTexture>("../../../../data/earthmap.jpeg");
    objects.add(make_shared<Sphere>(Point3(0,-1000,0), 1000, make_shared<Lambertian>(earthTexture)));
    objects.add(make_shared<Sphere>(Point3(0,2,0), 2, make_shared<Lambertian>(earthTexture)));

    auto DiffuseLightMat = make_shared<DiffuseLight>(Color(4,4,4));
    objects.add(make_shared<XYRectangle>(3, 5, 1, 3, -2, DiffuseLightMat));

    //auto GlassMat = make_shared<Dielectric>(1.5);
    // objects.add(make_shared<XYRectangle>(-100, 100, 0, 100, -2, GlassMat));

    return objects;
}

HittableList generateCornellBox() {
    HittableList objects;

    auto red   = make_shared<Lambertian>(Color(.65, .05, .05));
    auto white = make_shared<Lambertian>(Color(.73, .73, .73));
    auto green = make_shared<Lambertian>(Color(.12, .45, .15));
    auto light = make_shared<DiffuseLight>(Color(15, 15, 15));

    objects.add(make_shared<YZRectangle>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<YZRectangle>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<XZRectangle>(213, 343, 227, 332, 554, light));
    objects.add(make_shared<XZRectangle>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<XZRectangle>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<XYRectangle>(0, 555, 0, 555, 555, white));

    return objects;
}

int main() {
    // configure OpenMP
    omp_set_num_threads(8);

    // configure output image
    auto AspectRatio = 16.0 / 9.0;
    int ImageWidth = 400;
    int SamplesPerPixel = 100;
    int MaxRecursion = 8;

    // set Camera
    Point3 LookFrom;
    Point3 LookAt;
    Vector3 UpVector;
    double VerticalFOV = 40.0;
    double Aperture = 0.0;
    Color BackgroundColor(0.0, 0.0, 0.0);

    // set world
    HittableList World;
    int SceneSelector = 6;

    switch (SceneSelector) {
        case 0:
            World = generateRandomScene();
            BackgroundColor = Color(0.7, 0.8, 1.0);
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            Aperture = 0.1;
            break;

        case 1:
            World = generateRandomSceneMotionBlur();
            BackgroundColor = Color(0.7, 0.8, 1.0);
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            Aperture = 0.1;
            break;

        case 2:
            World = generateTwoSpheres();
            BackgroundColor = Color(0.7, 0.8, 1.0);
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            break;

        case 4:
            World = generateEarth();
            BackgroundColor = Color(0.7, 0.8, 1.0);
            LookFrom = Point3(13, 2, 3);
            LookAt = Point3(0, 0, 0);
            VerticalFOV = 20.0;
            break;

        case 5:
            World = generateSimpleLight();
            SamplesPerPixel = 400;
            BackgroundColor = Color(0,0,0);
            LookFrom = Point3(26,3,6);
            LookAt = Point3(0,2,0);
            VerticalFOV = 20.0;
            break;

        default:
        case 6:
            World = generateCornellBox();
            AspectRatio = 1.0;
            ImageWidth = 600;
            SamplesPerPixel = 200;
            BackgroundColor = Color(0,0,0);
            LookFrom = Point3(278, 278, -800);
            LookAt = Point3(278, 278, 0);
            VerticalFOV = 40.0;
            break;
    }

    int ImageHeight = static_cast<int>(ImageWidth / AspectRatio);
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

    #pragma omp parallel default(none) firstprivate(ImageHeight, ImageWidth) shared(cam, World, BackgroundColor, ImageBuffer, SamplesPerPixel, MaxRecursion)
    {
        #pragma omp for // trace rays & compute pixel colors
        for (int j = ImageHeight - 1; j >= 0; --j) {
            for (int i = 0; i < ImageWidth; ++i) {
                Color PixelColor(0, 0, 0);
                for (int s = 0; s < SamplesPerPixel; ++s) {
                    auto u = (i + generateRandomDouble()) / (ImageWidth - 1);
                    auto v = (j + generateRandomDouble()) / (ImageHeight - 1);
                    Ray r = cam.getRay(u, v);
                    PixelColor += computeRayColor(r, BackgroundColor, World, MaxRecursion);
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
