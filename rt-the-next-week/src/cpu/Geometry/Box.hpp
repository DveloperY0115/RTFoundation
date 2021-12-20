//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_BOX_HPP
#define RTFOUNDATION_BOX_HPP

#include "../rtweekend.hpp"
#include "AABBRectangle.hpp"
#include "HittableList.hpp"

class Box : public Hittable {
public:
    Box() {}
    Box(const Point3& Point0, const Point3& Point1, shared_ptr<Material> MaterialPtr) {
        BoxMinPoint = Point0;
        BoxMaxPoint = Point1;

        Sides.add(make_shared<XYRectangle>(Point0.X(), Point1.X(), Point0.Y(), Point1.Y(), Point1.Z(), MaterialPtr));
        Sides.add(make_shared<XYRectangle>(Point0.X(), Point1.X(), Point0.Y(), Point1.Y(), Point0.Z(), MaterialPtr));

        Sides.add(make_shared<XZRectangle>(Point0.X(), Point1.X(), Point0.Z(), Point1.Z(), Point1.Y(), MaterialPtr));
        Sides.add(make_shared<XZRectangle>(Point0.X(), Point1.X(), Point0.Z(), Point1.Z(), Point0.Y(), MaterialPtr));

        Sides.add(make_shared<YZRectangle>(Point0.Y(), Point1.Y(), Point0.Z(), Point1.Z(), Point1.X(), MaterialPtr));
        Sides.add(make_shared<YZRectangle>(Point0.Y(), Point1.Y(), Point0.Z(), Point1.Z(), Point0.X(), MaterialPtr));
    }

    bool hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const override {
        return Sides.hit(Ray, DepthMin, DepthMax, Record);
    }

    bool computeBoundingBox(double time0, double time1, AABB& output_box) const override {
        output_box = AABB(BoxMinPoint, BoxMaxPoint);
        return true;
    }


public:
    Point3 BoxMinPoint, BoxMaxPoint;
    HittableList Sides;
};
#endif //RTFOUNDATION_BOX_HPP
