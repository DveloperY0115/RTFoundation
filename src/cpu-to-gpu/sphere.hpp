//
// Created by 유승우 on 2020/04/27.
//

#ifndef FIRSTRAYTRACER_SPHERE_HPP
#define FIRSTRAYTRACER_SPHERE_HPP

#include "hittable.hpp"
#include "vector3.hpp"

/**
 * Sphere class which defines a sphere object that can interact with rays in the scene
 */
class sphere: public hittable
{
public:
    sphere() {}
    sphere(point3 cen, double r, shared_ptr<material> m)
    : center(cen), radius(r), mat_ptr(m) {};

    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;

    // member variables of 'sphere'
public:
    vector3 center;
    double radius;
    shared_ptr<material> mat_ptr;
};

/**
 * Determines whether given ray meets the surface of the caller
 * @param r a ray which will be tested
 * @param t_min the lower bound for ray offset 't'
 * @param t_max the upper bound for ray offset 't'
 * @param rec a structure to store information of the intersection (if it's turned out to be meaningful)
 * @return true if ray intersects with the surface, false otherwise
 */
bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    vector3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot_product(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b * half_b - a*c;

    // sphere is hit by the ray if and only if the equation has real solutions
    if (discriminant > 0)
    {
        // Solve for the solution that contains the actual parameter to get the point.
        auto root = sqrt(discriminant);

        // try smaller 't' first
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

        // try larger 't' then
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
