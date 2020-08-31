//
// Created by 유승우 on 2020/04/27.
//

#ifndef FIRSTRAYTRACER_SPHERE_HPP
#define FIRSTRAYTRACER_SPHERE_HPP

#include "hittable.hpp"
#include "vector3.hpp"

class sphere: public hittable
{
    /*
     * Note that sphere class inherits only the 'hit' function in this class.
     * The use of hit_record is allowed because the header is included.
     */
public:
    sphere() {}
    sphere(point3 cen, double r, shared_ptr<material> m)
    : center(cen), radius(r), mat_ptr(m) {};

    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;

public:
    vector3 center;
    double radius;
    shared_ptr<material> mat_ptr;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    vector3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot_product(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b * half_b - a*c;

    // sphere is hit by the ray
    if (discriminant > 0)
    {
        // Solve for the solution that contains the actual parameter to get the point.
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            // the point of the surface that was hit by the ray
            rec.p = r.at(rec.t);
            // here, we define a normal vector to point outward
            vector3 outward_normal = (rec.p - center) / radius;
            // compare the direction of the ray & outward_normal
            // set the normal, opposite to the direction where light came from
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }

        temp = (-half_b + root) / a;

        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vector3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }

    return false;
}

#endif //FIRSTRAYTRACER_SPHERE_HPP
