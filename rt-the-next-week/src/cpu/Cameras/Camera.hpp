//
// Created by 유승우 on 2020/05/15.
//

#ifndef RTFOUNDATION_CAMERA_HPP
#define RTFOUNDATION_CAMERA_HPP

#include "../rtweekend.hpp"

class Camera
{
public:
    Camera(
            Point3 LookFrom,
            Point3 LookAt,
            Vector3 UpVector,
            double VerticalFOV, // Vertical field-of-view in degrees
            double AspectRatio,
            double Aperture,
            double FocusingDistance,
            double ShutterOpenAt = 0,
            double ShutterCloseAt = 0
            ) {
            auto FOVTheta = degreeToRadian(VerticalFOV);
            auto h = tan(FOVTheta / 2);
            auto ViewportHeight = 2.0 * h;
            auto ViewportWidth = AspectRatio * ViewportHeight;

            // Set camera coordinate system
            w = normalize(LookFrom - LookAt);
            u = normalize(crossProduct(UpVector, w));
            v = crossProduct(w, u);

            Origin = LookFrom;
            Horizontal = FocusingDistance * ViewportWidth * u;
            Vertical = FocusingDistance * ViewportHeight * v;
            lowerLeftCorner = Origin - Horizontal / 2 - Vertical / 2 - FocusingDistance * w;

            LensRadius = Aperture / 2;

            // Set shutter time(s)
            ShutterOpenTime = ShutterOpenAt;
            ShutterCloseTime = ShutterCloseAt;
    }

    Ray getRay(double s, double t) const
    {
        Vector3 rd = LensRadius * randomInUnitDisk();
        Vector3 offset = u * rd.X() + v * rd.Y();

        return Ray(
                Origin + offset,
                lowerLeftCorner + s * Horizontal + t * Vertical - Origin - offset,
                generateRandomDouble(ShutterOpenTime, ShutterCloseTime)
                );
    }

private:
    Point3 Origin;
    Point3 lowerLeftCorner;
    Vector3 u, v, w;
    Vector3 Horizontal;
    Vector3 Vertical;
    double LensRadius;
    double ShutterOpenTime;
    double ShutterCloseTime;
};

#endif //RTFOUNDATION_CAMERA_HPP
