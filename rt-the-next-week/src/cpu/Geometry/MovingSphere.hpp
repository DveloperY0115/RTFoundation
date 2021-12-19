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
                 shared_ptr<Material> Material)
                 : CenterStart(CenterStart), CenterEnd(CenterEnd),
                   MovementStartTime(StartMovingAt), MovementEndTime(StopMovingAt),
                   Radius(Radius),
                   Material(Material)
    {
        // Do nothing
    };

    virtual bool Hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const override;

    Point3 CenterPositionAt(double Time) const;

private:
    Point3 CenterStart, CenterEnd;
    double MovementStartTime, MovementEndTime;
    double Radius;
    shared_ptr<Material> Material;
};

Point3 MovingSphere::CenterPositionAt(double Time) const {
    return CenterStart + ((Time - MovementStartTime) / (MovementEndTime - MovementStartTime)) * (CenterEnd - CenterStart);
}

#endif //RTFOUNDATION_MOVINGSPHERE_HPP
