//
// Created by 유승우 on 2020/04/27.
//

#ifndef RTFOUNDATION_HITTABLE_HPP
#define RTFOUNDATION_HITTABLE_HPP

#include "../rtweekend.hpp"
#include "../Rays/Ray.hpp"

class Material;

/*
 * The structure HitRecord stores:
 * i) The coordinate of hit-point
 * ii) The HitPointNormal vector of geometry getPointAt the Hit-point
 * iii) Material data of that point(or surface)
 * iv) Solution that gives the parameter to that point from the getRayOrigin
 * v) The boolean function that determines the relative getRayDirection of Ray and HitPointNormal
 */
struct HitRecord
{
public:
    inline void setFaceNormal(const Ray& Ray, const Vector3& OutwardNormal)
    {
        // front face is set to be true, if the HitPointNormal of the surface and incident Ray are opposite
        IsFrontFace = dotProduct(Ray.getRayDirection(), OutwardNormal) < 0;
        HitPointNormal = IsFrontFace ? OutwardNormal : -OutwardNormal;
    }

    Point3 HitPoint;
    Vector3 HitPointNormal;
    shared_ptr<Material> MaterialPtr;
    double Depth;
    bool IsFrontFace;
};


class Hittable
{
public:
    // virtual : the member function that is expected to be re-define in a derived class
    virtual bool Hit(const Ray& Ray, double DepthMin, double DepthMax, HitRecord& Record) const = 0;
};

#endif //RTFOUNDATION_HITTABLE_HPP
