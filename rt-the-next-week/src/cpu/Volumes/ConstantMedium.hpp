//
// Created by 유승우 on 2021/12/20.
//

#ifndef RTFOUNDATION_CONSTANTMEDIUM_HPP
#define RTFOUNDATION_CONSTANTMEDIUM_HPP

#include "../rtweekend.hpp"
#include "../Geometry/Hittable.hpp"
#include "../Materials/Material.hpp"
#include "../Materials/Isotropic.hpp"
#include "../Textures/Texture.hpp"

class ConstantMedium : public Hittable {
public:
    ConstantMedium(shared_ptr<Hittable> Boundary_, double Density_, shared_ptr<Texture> MediumTexture_)
    : Boundary(Boundary_),
    NegativeInverseDensity(-1 / Density_),
    PhaseFunction(make_shared<Isotropic>(MediumTexture_))
    {
        // Do nothing.
    }

    ConstantMedium(shared_ptr<Hittable> Boundary_, double Density_, Color MediumTextureColor_)
    : Boundary(Boundary_),
    NegativeInverseDensity(-1 / Density_),
    PhaseFunction(make_shared<Isotropic>(MediumTextureColor_))
    {
        // Do nothing.
    }

    bool hit(const Ray& IncidentRay, double DepthMin, double DepthMax, HitRecord& Record) const override {
        HitRecord Record0, Record1;

        if (!Boundary->hit(IncidentRay, -Infinity, Infinity, Record0))
            return false;

        if (!Boundary->hit(IncidentRay, Record0.Depth + 0.0001, Infinity, Record1))
            return false;

        if (Record0.Depth < DepthMin)
            Record0.Depth = DepthMin;
        if (Record1.Depth > DepthMax)
            Record1.Depth = DepthMax;

        if (Record0.Depth >= Record1.Depth)
            return false;

        if (Record0.Depth < 0)
            Record0.Depth = 0;

        const double RayLength = IncidentRay.getRayDirection().length();
        const double DistanceInsideBoundary = (Record1.Depth - Record0.Depth) * RayLength;
        const double HitDistance = NegativeInverseDensity * log(generateRandomDouble());

        if (HitDistance > DistanceInsideBoundary)
            return false;

        Record.Depth = Record0.Depth + HitDistance / RayLength;
        Record.HitPoint = IncidentRay.getPointAt(Record.Depth);

        Record.HitPointNormal = Vector3(1, 0, 0);
        Record.IsFrontFace = true;
        Record.MaterialPtr = PhaseFunction;

        return true;
    }

    bool computeBoundingBox(double Time0, double Time1, AABB& OutputBoundingBox) const override {
        return Boundary->computeBoundingBox(Time0, Time1, OutputBoundingBox);
    }

public:
    shared_ptr<Hittable> Boundary;
    double NegativeInverseDensity;
    shared_ptr<Material> PhaseFunction;
};
#endif //RTFOUNDATION_CONSTANTMEDIUM_HPP
