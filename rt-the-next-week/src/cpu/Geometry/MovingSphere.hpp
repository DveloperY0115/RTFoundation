//
// Created by 유승우 on 2021/12/19.
//

#ifndef RTFOUNDATION_MOVINGSPHERE_HPP
#define RTFOUNDATION_MOVINGSPHERE_HPP

#include "../rtweekend.hpp"

#include "Hittable.hpp"

class MovingSphere : public Hittable {
public:
    MovingSphere() {}
    MovingSphere(Point3 CenterStart,
                 Point3 CenterEnd,
                 double StartMovingAt,
                 double StopMovingAt,
                 double Radius,
                 shared_ptr<Material> MaterialPtr)
                 : CenterStart(CenterStart), CenterEnd(CenterEnd),
                   MovementStartTime(StartMovingAt), MovementEndTime(StopMovingAt),
                   Radius(Radius),
                   MaterialPtr(MaterialPtr)
    {
        // Do nothing
    };

    bool hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const override;

    Point3 CenterPositionAt(double Time) const;

private:
    Point3 CenterStart, CenterEnd;
    double MovementStartTime, MovementEndTime;
    double Radius;
    shared_ptr<Material> MaterialPtr;
};

Point3 MovingSphere::CenterPositionAt(double Time) const {
    return CenterStart + ((Time - MovementStartTime) / (MovementEndTime - MovementStartTime)) * (CenterEnd - CenterStart);
}

bool MovingSphere::hit(const Ray &Ray, double DepthMin, double DepthMax, HitRecord &Record) const {
    Vector3 oc = Ray.getRayOrigin() - CenterPositionAt(Ray.getCreatedTime());
    auto a = Ray.getRayDirection().lengthSquared();
    auto half_b = dotProduct(oc, Ray.getRayDirection());
    auto c = oc.lengthSquared() - Radius * Radius;

    auto discriminant = half_b * half_b - a*c;

    // Sphere is hit by the Ray if and only if the equation has real solutions
    if (discriminant > 0)
    {
        // Solve for the solution that contains the actual parameter to get the point.
        auto root = sqrt(discriminant);

        // try smaller 'Depth' first
        auto temp = (-half_b - root) / a;
        if (temp < DepthMax && temp > DepthMin)
        {
            Record.Depth = temp;
            // the point of the surface that was hit by the Ray
            Record.HitPoint = Ray.getPointAt(Record.Depth);
            // here, we define a HitPointNormal vector to point outward
            Vector3 OutwardNormal = (Record.HitPoint - CenterPositionAt(Ray.getCreatedTime())) / Radius;
            // compare the getRayDirection of the Ray & OutwardNormal
            // set the HitPointNormal, opposite to the getRayDirection where light came from
            Record.setFaceNormal(Ray, OutwardNormal);
            Record.MaterialPtr = MaterialPtr;
            return true;
        }

        // try larger 'Depth' then
        temp = (-half_b + root) / a;

        if (temp < DepthMax && temp > DepthMin)
        {
            Record.Depth = temp;
            Record.HitPoint = Ray.getPointAt(Record.Depth);
            Vector3 OutwardNormal = (Record.HitPoint - CenterPositionAt(Ray.getCreatedTime())) / Radius;
            Record.setFaceNormal(Ray, OutwardNormal);
            Record.MaterialPtr = MaterialPtr;
            return true;
        }
    }

    return false;
}

#endif //RTFOUNDATION_MOVINGSPHERE_HPP
