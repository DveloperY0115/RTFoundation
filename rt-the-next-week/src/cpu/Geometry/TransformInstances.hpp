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
        // Inversely translate the ray.
        Ray TranslatedRay = Ray(IncidentRay.getRayOrigin() - Displacement, IncidentRay.getRayDirection(), IncidentRay.getCreatedTime());
        if (!TranslatedObject->hit(TranslatedRay, DepthMin, DepthMax, Record)) {
            return false;
        }

        // Translate the hit point along the original direction.
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

class YRotationInstance : public Hittable {
public:
    YRotationInstance() {}
    YRotationInstance(shared_ptr<Hittable> RotatedObject_, double AngleInDegree_) : RotatedObject(RotatedObject_) {
        double AngleInRadian = degreeToRadian(AngleInDegree_);
        SineTheta = sin(AngleInRadian);
        CosineTheta = cos(AngleInRadian);
        hasBox = RotatedObject_->computeBoundingBox(0, 1, BoundingBox);

        Point3 Min = Point3(Infinity, Infinity, Infinity);
        Point3 Max = Point3(-Infinity, -Infinity, -Infinity);

        // TODO: Simplify this using the matrix multiplication!
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    double x = i*BoundingBox.getMax().X() + (1-i)*BoundingBox.getMin().X();
                    double y = j*BoundingBox.getMax().Y() + (1-j)*BoundingBox.getMin().Y();
                    double z = k*BoundingBox.getMax().Z() + (1-k)*BoundingBox.getMin().Z();

                    double NewX = CosineTheta * x + SineTheta * z;
                    double NewZ = -SineTheta * x + CosineTheta * z;

                    Vector3 Tester(NewX, y, NewZ);

                    for (int c = 0; c < 3; c++) {
                        Min[c] = fmin(Min[c], Tester[c]);
                        Max[c] = fmax(Max[c], Tester[c]);
                    }
                }
            }
        }

        BoundingBox = AABB(Min, Max);
    }

    bool hit(const Ray& IncidentRay, double DepthMin, double DepthMax, HitRecord& Record) const override {
        Point3 RotatedRayOrigin = IncidentRay.getRayOrigin();
        Vector3 RotatedRayDirection = IncidentRay.getRayDirection();

        // Inversely rotate the ray origin and direction, respectively.
        RotatedRayOrigin[0] = CosineTheta * IncidentRay.getRayOrigin()[0] - SineTheta * IncidentRay.getRayOrigin()[2];
        RotatedRayOrigin[2] = SineTheta * IncidentRay.getRayOrigin()[0] + CosineTheta * IncidentRay.getRayOrigin()[2];
        RotatedRayDirection[0] = CosineTheta * IncidentRay.getRayDirection()[0] - SineTheta * IncidentRay.getRayDirection()[2];
        RotatedRayDirection[2] = SineTheta * IncidentRay.getRayDirection()[0] + CosineTheta * IncidentRay.getRayDirection()[2];

        Ray RotatedRay = Ray(RotatedRayOrigin, RotatedRayDirection, IncidentRay.getCreatedTime());

        if (!RotatedObject->hit(RotatedRay, DepthMin, DepthMax, Record))
            return false;

        // Rotate the hit point and the surface normal at that point along the original direction.
        Point3 RotatedHitPoint = Record.HitPoint;
        Vector3 RotatedHitPointNormal = Record.HitPointNormal;

        /*
         * Note that the rotation matrix around the Y-axis is:
         * | Cos    0    Sin |
         * |  0     1     0  |
         * | -Sin   0    Cos |
         */
        RotatedHitPoint[0] = CosineTheta * Record.HitPoint[0] + SineTheta * Record.HitPoint[2];
        RotatedHitPoint[2] = -SineTheta * Record.HitPoint[0] + CosineTheta * Record.HitPoint[2];
        RotatedHitPointNormal[0] = CosineTheta * Record.HitPointNormal[0] + SineTheta * Record.HitPointNormal[2];
        RotatedHitPointNormal[2] = -SineTheta * Record.HitPointNormal[0] + CosineTheta * Record.HitPointNormal[2];

        Record.HitPoint = RotatedHitPoint;
        Record.setFaceNormal(RotatedRay, RotatedHitPointNormal);

        return true;
    }

    bool computeBoundingBox(double t0, double t1, AABB& OutputBoundingBox) const override {
        OutputBoundingBox = BoundingBox;
        return hasBox;
    }

public:
    shared_ptr<Hittable> RotatedObject;
    double SineTheta, CosineTheta;
    bool hasBox;
    AABB BoundingBox;
};

#endif //RTFOUNDATION_TRANSFORMINSTANCES_HPP
