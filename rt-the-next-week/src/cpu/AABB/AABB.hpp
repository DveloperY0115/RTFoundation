//
// Created by 유승우 on 2021/12/19.
//

#ifndef RTFOUNDATION_AABB_HPP
#define RTFOUNDATION_AABB_HPP

#include "../rtweekend.hpp"

class AABB {
    AABB() {}
    AABB(const Point3& A, const Point3& B)
    : Minimum(A), Maximum(B)
    {
        // Do nothing
    }

    Point3 getMin() const {
        return Minimum;
    }

    Point3 getMax() const {
        return Maximum;
    }

    bool hit(const Ray& IncidentRay, double DepthMin, double DepthMax) const {
        for (int i = 0; i < 3; ++i) {
            double invB = 1.0f / IncidentRay.getRayDirection()[i];
            double t0 = (getMin()[i] - IncidentRay.getRayOrigin()[i]) * invB;    // (A_min_i - O_i) / d_i
            double t1 = (getMax()[i] - IncidentRay.getRayOrigin()[i]) * invB;    // (A_max_i - O_i) / d_i
            if (invB < 0.0f)
                std::swap(t0, t1);
            DepthMin = t0 > DepthMin ? t0 : DepthMin;
            DepthMax = t1 < DepthMax ? t1 : DepthMax;
            if (DepthMin >= DepthMax)
                return false;
        }
        return true;
    }

public:
    Point3 Minimum, Maximum;
};

#endif //RTFOUNDATION_AABB_HPP
