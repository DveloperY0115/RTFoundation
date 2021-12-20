//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_AABBRECTANGLE_HPP
#define RTFOUNDATION_AABBRECTANGLE_HPP

#include "../rtweekend.hpp"
#include "../Geometry/Hittable.hpp"

class XYRectangle : public Hittable {
public:
    XYRectangle() {}
    XYRectangle(double x0_, double x1_, double y0_, double y1_, double z_, shared_ptr<Material> RectangleMaterial_)
    : x0(x0_), x1(x1_), y0(y0_), y1(y1_), z(z_), RectangleMaterial(RectangleMaterial_)
    {
        // Do nothing.
    }

    bool hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const override {
        auto HitDepth = (z - Ray.getRayOrigin().Z()) / Ray.getRayDirection().Z();

        if (HitDepth < DepthMin || HitDepth > DepthMax)
            return false;

        auto PointOnRayAtHitDepth = Ray.getPointAt(HitDepth);
        auto x = PointOnRayAtHitDepth.X();
        auto y = PointOnRayAtHitDepth.Y();
        if (x < x0 || x > x1 || y < y0 || y > y1)
            return false;
        Record.u = (x - x0) / (x1 - x0);
        Record.v = (y - y0) / (y1 - y0);
        Record.Depth = HitDepth;

        auto OutwardNormal = Vector3(0, 0, 1);
        Record.setFaceNormal(Ray, OutwardNormal);
        Record.MaterialPtr = RectangleMaterial;
        Record.HitPoint = PointOnRayAtHitDepth;
        return true;
    }

    virtual bool computeBoundingBox(double t0, double t1, AABB& OutputBoundingBox) const override {
        OutputBoundingBox = AABB(Point3(x0, y0, z-0.0001), Point3(x1, y1, z+0.0001));
        return true;
    }

public:
    double x0, x1, y0, y1, z;
    shared_ptr<Material> RectangleMaterial;
};

#endif //RTFOUNDATION_AABBRECTANGLE_HPP
