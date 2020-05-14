//
// Created by 유승우 on 2020/04/27.
//

#ifndef FIRSTRAYTRACER_SPHERE_H
#define FIRSTRAYTRACER_SPHERE_H

#include "hittable/hittable.h"
#include "vector3/vector3.h"

class sphere: public hittable
{
public:
    sphere() {}
    sphere(point3 cen, double r) : center(cen), radius(r) {};

    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;

public:
    vector3 center;
    double radius;
};

#endif //FIRSTRAYTRACER_SPHERE_H
