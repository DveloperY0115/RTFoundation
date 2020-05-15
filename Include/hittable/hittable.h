//
// Created by 유승우 on 2020/04/27.
//

#ifndef FIRSTRAYTRACER_HITTABLE_H
#define FIRSTRAYTRACER_HITTABLE_H

#include "../Include/ray/ray.h"

struct hit_record
{
    point3 p;
    vector3 normal;
    double t;
    bool front_face;

    inline void set_face_normal(const ray& r, const vector3& outward_normal)
    {
        // if the direction of light and the outward directed normal vector is opposite,
        // this function returns true
        front_face = dot_product(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }

};


class hittable
{
public:
    // virtual : the member function that is expected to be redefine in a derived class
    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif //FIRSTRAYTRACER_HITTABLE_H
