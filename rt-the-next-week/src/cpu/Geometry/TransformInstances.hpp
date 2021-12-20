//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_TRANSFORMINSTANCES_HPP
#define RTFOUNDATION_TRANSFORMINSTANCES_HPP

#include "Hittable.hpp"

class TranslateInstance : public Hittable {
public:
    TranslateInstance(shared_ptr<Hittable> TranslatedObject_, const Vector3& Displacement_)
    : TranslatedObject(TranslatedObject_), Displacement(Displacement_)
    {
        // Do nothing.
    }

    bool hit(const Ray& IncidentRay, double DepthMin, double DepthMax, HitRecord& Record) const override {
        Ray TranslatedRay = Ray(IncidentRay.getRayOrigin() - Displacement, IncidentRay.getRayDirection(), IncidentRay.getCreatedTime());
        if (!TranslatedObject->hit(TranslatedRay, DepthMin, DepthMax, Record)) {
            return false;
        }

        Record.HitPoint += Displacement;
        Record.setFaceNormal(TranslatedRay, Record.HitPointNormal);

        return true;
    }

    bool computeBoundingBox(double t0, double t1, AABB& OutputBoundingBox) const override {
        if (!TranslatedObject->computeBoundingBox(t0, t1, OutputBoundingBox))
            return false;

        OutputBoundingBox = AABB(
                OutputBoundingBox.getMin() + Displacement,
                OutputBoundingBox.getMax() + Displacement
                );

        return true;
    }

public:
    shared_ptr<Hittable> TranslatedObject;
    Vector3 Displacement;
};

#endif //RTFOUNDATION_TRANSFORMINSTANCES_HPP
