//
// Created by dveloperY0115 on 12/29/2020.
//

#ifndef RTFOUNDATION_DIELECTRIC_H
#define RTFOUNDATION_DIELECTRIC_H

#include "Material.hpp"

/*
 * Dielectric Material class
 */
class Dielectric: public Material {
public:
    Dielectric(double IndexOfRefraction) : ir(IndexOfRefraction) {}

    virtual bool scatter(
            const Ray& IncidentRay, const HitRecord& Record, Color& Attenuation, Ray& ScatteredRay
    ) const override {
        Attenuation = Color(1.0, 1.0, 1.0);
        double RefractionRatio = Record.IsFrontFace ? (1.0 / ir) : ir;   // if incident Ray is from the inside of Dielectric, (ratio of IOR) = (index of Dielectric) / 1.0 (air)

        Vector3 incidentDirection = normalize(IncidentRay.getRayDirection());
        double cosTheta = fmin(dotProduct(-incidentDirection, Record.HitPointNormal), 1.0);
        double sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = RefractionRatio * sinTheta > 1.0;
        Vector3 refractedDirection;

        if (cannotRefract || reflectance(cosTheta, RefractionRatio) > generateRandomDouble()) {
            refractedDirection = reflect(incidentDirection, Record.HitPointNormal);
        } else {
            refractedDirection = refract(incidentDirection, Record.HitPointNormal, RefractionRatio);
        }

        ScatteredRay = Ray(Record.HitPoint, refractedDirection);
        return true;
    }

public:
    double ir;  // index of refraction

private:
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif //RTFOUNDATION_DIELECTRIC_H
