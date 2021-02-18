#ifndef RAY_TRACING_IN_CPP_MOVING_SPHERE_H
#define RAY_TRACING_IN_CPP_MOVING_SPHERE_H

#include "hittable.hpp"

class moving_sphere: public hittable  {
public:
    __device__ moving_sphere() {}

    __device__ moving_sphere(
            vector3 cen0, vector3 cen1, float _time0, float _time1, float r, material* m
                             ) : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m)  {};

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

    __device__ vector3 center(float time) const;

public:
    vector3 center0, center1;
    float radius;
    material* mat_ptr;
    float time0, time1;
};

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vector3 oc = r.origin() - center(r.time());
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {

        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.set_face_normal(r, rec.normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center(r.time())) / radius;
            rec.set_face_normal(r, rec.normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ vector3 moving_sphere::center(float time) const {
    return  center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}
#endif //RAY_TRACING_IN_CPP_MOVING_SPHERE_H
