#include <array>
#include <iostream>
#include <omp.h>

#include "Cameras/Camera.hpp"
#include "rtweekend.hpp"
#include "Geometry/HittableList.hpp"
#include "Geometry/Sphere.hpp"
#include "Geometry/AABBRectangle.hpp"
#include "Geometry/TransformInstances.hpp"
#include "Geometry/Box.hpp"
#include "Geometry/MovingSphere.hpp"
#include "Geometry/BVH.hpp"
#include "Colors/Colors.hpp"
#include "Materials/Material.hpp"
#include "Materials/Lambertian.hpp"
#include "Materials/Metal.hpp"
#include "Materials/Dielectric.hpp"
#include "Materials/DiffuseLight.hpp"
#include "Textures/CheckerTexture.hpp"
#include "Textures/ImageTexture.hpp"
#include "Volumes/ConstantMedium.hpp"

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

    auto checker = make_shared<CheckerTexture>(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9), 10.0);
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

    auto Checker = make_shared<CheckerTexture>(Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9), 10.0);

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

    shared_ptr<Hittable> Box1 = make_shared<Box>(Point3(0, 0, 0), Point3(165, 330, 165), white);
    Box1 = make_shared<YRotationInstance>(Box1, 15);
    Box1 = make_shared<TranslateInstance>(Box1, Vector3(265,0,295));
    objects.add(Box1);

    shared_ptr<Hittable> Box2 = make_shared<Box>(Point3(0,0,0), Point3(165,165,165), white);
    Box2 = make_shared<YRotationInstance>(Box2, -18);
    Box2 = make_shared<TranslateInstance>(Box2, Vector3(130,0,65));
    objects.add(Box2);

    return objects;
}

HittableList generateCornellSmoke() {
    HittableList objects;

    auto red   = make_shared<Lambertian>(Color(.65, .05, .05));
    auto white = make_shared<Lambertian>(Color(.73, .73, .73));
    auto green = make_shared<Lambertian>(Color(.12, .45, .15));
    auto light = make_shared<DiffuseLight>(Color(7, 7, 7));

    objects.add(make_shared<YZRectangle>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<YZRectangle>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<XZRectangle>(113, 443, 127, 432, 554, light));
    objects.add(make_shared<XZRectangle>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<XZRectangle>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<XYRectangle>(0, 555, 0, 555, 555, white));

    shared_ptr<Hittable> box1 = make_shared<Box>(Point3(0,0,0), Point3(165,330,165), white);
    box1 = make_shared<YRotationInstance>(box1, 15);
    box1 = make_shared<TranslateInstance>(box1, Vector3(265,0,295));

    shared_ptr<Hittable> box2 = make_shared<Box>(Point3(0,0,0), Point3(165,165,165), white);
    box2 = make_shared<YRotationInstance>(box2, -18);
    box2 = make_shared<TranslateInstance>(box2, Vector3(130,0,65));

    objects.add(make_shared<ConstantMedium>(box1, 0.01, Color(0,0,0)));
    objects.add(make_shared<ConstantMedium>(box2, 0.01, Color(1,1,1)));

    return objects;
}

HittableList generateFinalScene() {
    HittableList boxes1;
    auto ground = make_shared<Lambertian>(Color(0.48, 0.83, 0.53));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = generateRandomDouble(1, 101);
            auto z1 = z0 + w;

            boxes1.add(make_shared<Box>(Point3(x0, y0, z0), Point3(x1, y1, z1), ground));
        }
    }

    HittableList objects;

    objects.add(make_shared<BVHNode>(boxes1, 0, 1));

    auto light = make_shared<DiffuseLight>(Color(7, 7, 7));
    objects.add(make_shared<XZRectangle>(123, 423, 147, 412, 554, light));

    auto center1 = Point3(400, 400, 200);
    auto center2 = center1 + Vector3(30, 0, 0);
    auto moving_sphere_material = make_shared<Lambertian>(Color(0.7, 0.3, 0.1));
    objects.add(make_shared<MovingSphere>(center1, center2, 0, 1, 50, moving_sphere_material));

    objects.add(make_shared<Sphere>(Point3(260, 150, 45), 50, make_shared<Dielectric>(1.5)));
    objects.add(make_shared<Sphere>(
            Point3(0, 150, 145), 50, make_shared<Metal>(Color(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = make_shared<Sphere>(Point3(360,150,145), 70, make_shared<Dielectric>(1.5));
    objects.add(boundary);
    objects.add(make_shared<ConstantMedium>(boundary, 0.2, Color(0.2, 0.4, 0.9)));
    boundary = make_shared<Sphere>(Point3(0, 0, 0), 5000, make_shared<Dielectric>(1.5));
    objects.add(make_shared<ConstantMedium>(boundary, .0001, Color(1,1,1)));

    auto emat = make_shared<Lambertian>(make_shared<ImageTexture>("../../../../data/earthmap.jpeg"));
    objects.add(make_shared<Sphere>(Point3(400,200,400), 100, emat));
    // auto pertext = make_shared<noise_texture>(0.1);
    // objects.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

    HittableList boxes2;
    auto white = make_shared<Lambertian>(Color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<Sphere>(Point3::random(0,165), 10, white));
    }

    objects.add(make_shared<TranslateInstance>(
                        make_shared<YRotationInstance>(
                                make_shared<BVHNode>(boxes2, 0.0, 1.0), 15),
                        Vector3(-100,270,395)
                )
    );

    return objects;
}

HittableList generateWhitted() {
    HittableList Objects;

    // floor with red, yellow tile
    auto FloorCheckerTexture = make_shared<CheckerTexture>(Color(1.0, 0.0, 0.0), Color(1.0, 1.0, 0.0), 1.0);
    auto FloorMaterial = make_shared<Lambertian>(FloorCheckerTexture);
    shared_ptr<Hittable> Floor = make_shared<Sphere>(Point3(0, -1005, 0), 1000, FloorMaterial);
    Objects.add(Floor);

    // hollow glass ball
    auto GlassMaterial = make_shared<Dielectric>(1.5);
    shared_ptr<Hittable> GlassBallOuter = make_shared<Sphere>(Point3(7.5, 4.0, 6.0), 2.5, GlassMaterial);
    shared_ptr<Hittable> GlassBallInner = make_shared<Sphere>(Point3(7.5, 4.0, 6.0), -2.4, GlassMaterial);
    Objects.add(GlassBallOuter);
    Objects.add(GlassBallInner);

    // metal ball
    auto SilverMaterial = make_shared<Metal>(Color(192.0 / 255.0, 192.0 / 255.0, 192.0 / 255.0), 0);
    shared_ptr<Hittable> SilverBall = make_shared<Sphere>(Point3(5.0, 2.5, 0.0), 2.5, SilverMaterial);
    Objects.add(SilverBall);

    // point light source
    auto WhiteDiffuseLight = make_shared<DiffuseLight>(Color(15.0, 15.0, 15.0));
    shared_ptr<Hittable> PointLight = make_shared<Sphere>(Point3(70.0, 70.0, 70.0), 30.0, WhiteDiffuseLight);
    Objects.add(PointLight);

    return Objects;
}

int main() {
    // configure OpenMP
    omp_set_num_threads(8);

    // configure output image
    auto AspectRatio = 16.0 / 9.0;
    int ImageWidth = 400;
    int SamplesPerPixel = 100;
    int MaxRecursion = 8;

    // set time
    double TimeStart = 0.0;
    double TimeEnd = 0.0;

    // set Camera
    Point3 LookFrom;
    Point3 LookAt;
    Vector3 UpVector;
    double VerticalFOV = 40.0;
    double Aperture = 0.0;
    Color BackgroundColor(0.0, 0.0, 0.0);

    // set world
    BVHNode WorldBVH;
    HittableList World;
    int SceneSelector = 9;

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
            TimeEnd = 1.0;
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
            ImageWidth = 1600;
            SamplesPerPixel = 400;
            BackgroundColor = Color(0,0,0);
            LookFrom = Point3(26,3,6);
            LookAt = Point3(0,2,0);
            VerticalFOV = 20.0;
            break;

        case 6:
            World = generateCornellBox();
            AspectRatio = 1.0;
            ImageWidth = 600;
            SamplesPerPixel = 1000;
            BackgroundColor = Color(0,0,0);
            LookFrom = Point3(278, 278, -800);
            LookAt = Point3(278, 278, 0);
            VerticalFOV = 40.0;
            break;

        case 7:
            World = generateCornellSmoke();
            AspectRatio = 1.0;
            ImageWidth = 600;
            SamplesPerPixel = 400;
            LookFrom = Point3(278, 278, -800);
            LookAt = Point3(278, 278, 0);
            VerticalFOV = 40.0;
            break;

        default:
        case 8:
            World = generateFinalScene();
            AspectRatio = 1.0;
            ImageWidth = 800;
            SamplesPerPixel = 10000;
            BackgroundColor = Color(0,0,0);
            LookFrom = Point3(478, 278, -600);
            LookAt = Point3(278, 278, 0);
            VerticalFOV = 40.0;
            break;

        case 9:
            World = generateWhitted();
            AspectRatio = 1.0;
            ImageWidth = 800;
            SamplesPerPixel = 400;
            BackgroundColor = Color(0.7,0.8,1.0);
            LookFrom = Point3(7.5, 4.0, 20);
            LookAt = Point3(7.5, 4.0, 2.5);
            VerticalFOV = 40.0;
            break;
    }

    // Construct BVH using the HittableList instance
    // WorldBVH = BVHNode(World, TimeStart, TimeEnd);

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
