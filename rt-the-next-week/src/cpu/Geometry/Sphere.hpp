//
// Created by 유승우 on 2020/04/27.
//

#ifndef RTFOUNDATION_SPHERE_HPP
#define RTFOUNDATION_SPHERE_HPP

#include "Hittable.hpp"
#include "../Math/Vector3.hpp"

/**
 * Sphere class which defines a Sphere object that can interact with rays in the scene
 */
class Sphere: public Hittable
{
public:
    Sphere() {}
    Sphere(Point3 Center, double Radius, shared_ptr<Material> MaterialPtr)
    : Center(Center), Radius(Radius), MaterialPtr(MaterialPtr) {};

    bool hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const override;
    bool computeBoundingBox(double t0, double t1, AABB &OutputBoundingBox) const override;

private:
    static void computeSphereUV(const Point3& SurfacePoint, double& u, double& v) {
        auto Theta = acos(-SurfacePoint.Y());
        auto Phi = atan2(-SurfacePoint.Z(), SurfacePoint.X()) + Pi;

        u = Phi / (2 * Pi);
        v = Theta / Pi;
    }
public:
    Vector3 Center;
    double Radius;
    shared_ptr<Material> MaterialPtr;
};

/**
 * Determines whether given Ray meets the surface of the caller
 * @param Ray a Ray which will be tested
 * @param DepthMin the lower bound for Ray offset 'Depth'
 * @param DepthMax the upper bound for Ray offset 'Depth'
 * @param Record a structure to store information of the intersection (if it's turned out to be meaningful)
 * @return true if Ray intersects with the surface, false otherwise
 */
bool Sphere::hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const {
    Vector3 oc = Ray.getRayOrigin() - Center;
    auto a = Ray.getRayDirection().lengthSquared();
    auto half_b = dotProduct(oc, Ray.getRayDirection());
    auto c = oc.lengthSquared() - Radius * Radius;

    auto discriminant = half_b * half_b - a*c;

    // Sphere is hit by the Ray if and only if the equation has real solutions
    if (discriminant > 0) {
        // Solve for the solution that contains the actual parameter to get the point.
        auto root = sqrt(discriminant);

        // try smaller 'Depth' first
        auto temp = (-half_b - root) / a;
        if (temp < DepthMax && temp > DepthMin) {
            Record.Depth = temp;
            // the point of the surface that was hit by the Ray
            Record.HitPoint = Ray.getPointAt(Record.Depth);
            // here, we define a HitPointNormal vector to point outward
            Vector3 OutwardNormal = (Record.HitPoint - Center) / Radius;
            Record.setFaceNormal(Ray, OutwardNormal);
            computeSphereUV(OutwardNormal, Record.u, Record.v);
            Record.MaterialPtr = MaterialPtr;
            return true;
        }

        // try larger 'Depth' then
        temp = (-half_b + root) / a;

        if (temp < DepthMax && temp > DepthMin) {
            Record.Depth = temp;
            Record.HitPoint = Ray.getPointAt(Record.Depth);
            Vector3 OutwardNormal = (Record.HitPoint - Center) / Radius;
            Record.setFaceNormal(Ray, OutwardNormal);
            computeSphereUV(OutwardNormal, Record.u, Record.v);
            Record.MaterialPtr = MaterialPtr;
            return true;
        }
    }

    return false;
}

bool Sphere::computeBoundingBox(double t0, double t1, AABB& OutputBoundingBox) const {
    OutputBoundingBox = AABB(
            Center - Vector3(Radius, Radius, Radius),
            Center + Vector3(Radius, Radius, Radius)
            );
    return true;
}

#endif //RTFOUNDATION_SPHERE_HPP
