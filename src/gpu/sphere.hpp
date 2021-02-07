//
// Created by dveloperY0115 on 1/8/2021.
//

#ifndef RAY_TRACING_IN_CPP_SPHERE_H
#define RAY_TRACING_IN_CPP_SPHERE_H

#include "hittable.hpp"

class sphere: public hittable  {
public:
    __device__ sphere() {}
    __device__ sphere(vector3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m)  {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    vector3 center;
    float radius;
    material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vector3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {

        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.set_face_normal(r, rec.normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.set_face_normal(r, rec.normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif //RAY_TRACING_IN_CPP_SPHERE_H
