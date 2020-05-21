//
// Created by 유승우 on 2020/04/27.
//

#ifndef FIRSTRAYTRACER_HITTABLE_HPP
#define FIRSTRAYTRACER_HITTABLE_HPP

#include "rtweekend.hpp"
#include "ray.hpp"

class material;

/*
 * The structure hit_record stores:
 * i) The coordinate of hit-point
 * ii) The normal vector of geometry at the hit-point
 * iii) Material data of that point(or surface)
 * iv) Solution that gives the parameter to that point from the origin
 * v) The boolean function that determines the relative direction of ray and normal
 */
struct hit_record
{
    point3 p;
    vector3 normal;
    shared_ptr<material> mat_ptr;
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

#endif //FIRSTRAYTRACER_HITTABLE_HPP
