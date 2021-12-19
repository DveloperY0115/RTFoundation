//
// Created by 유승우 on 2020/04/21.
//

#ifndef RTFOUNDATION_VECTOR3_HPP
#define RTFOUNDATION_VECTOR3_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

class Vector3
{
public:
    /*
     * Empty constructor for 3D vector
     */
    Vector3() : e{0, 0, 0} {
        // Do nothing
    }

    Vector3(double x, double y, double z) {
        e[0] = x;
        e[1] = y;
        e[2] = z;
    }

    /*
     * Getters for vector elements.
     */
    double X() const { return e[0]; }
    double Y() const { return e[1]; }
    double Z() const { return e[2]; }

    /*
     * Setters for vector elements.
     */
    void setX(double x) { e[0] = x; }
    void setY(double y) { e[1] = y; }
    void setZ(double z) { e[2] = z; }

    /*
     * Unary operators
     */
    Vector3 operator - () const {
        return Vector3(-X(), -Y(), -Z());
    }

    double operator[](int Index) const {
        return e[Index];
    }

    double& operator[](int Index) {
        return e[Index];
    }

    /*
     * Binary operators
     */
    Vector3& operator += (const Vector3 &v) {
        e[0] += v.X();
        e[1] += v.Y();
        e[2] += v.Z();
        return *this;
    }

    Vector3& operator -= (const Vector3 &v) {
        e[0] -= v.X();
        e[1] -= v.Y();
        e[2] -= v.Z();
        return *this;
    }

    Vector3& operator *= (const Vector3 &v) {
        e[0] *= v.X();
        e[1] *= v.Y();
        e[2] *= v.Z();
        return *this;
    }

    Vector3& operator /= (const Vector3 &v) {
        e[0] /= v.X();
        e[1] /= v.Y();
        e[2] /= v.Z();
        return *this;
    }

    Vector3& operator *= (const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    Vector3& operator /= (const double t) {
        return *this *= 1/t;
    }

    /*
     * Compute the Euclidean norm of the calling vector.
     */
    double length() const {
        return sqrt(lengthSquared());
    }

    /*
     * Compute the squared sum of vector elements.
     */
    double lengthSquared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    /*
     * Normalize the calling vector
     */
    void normalize() {
        double k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] * e[2]*e[2]);
        e[0] *= k; e[1] *= k; e[2] *= k;
    }

    /*
     * Create a 3D vector whose elements are initialized with random numbers.
     */
    inline static Vector3 random() {
        return Vector3(
                generateRandomDouble(),
                generateRandomDouble(),
                generateRandomDouble()
                );
    }

    /*
     * Create a 3D vector whose elements are initialized with random numbers lying in the given interval.
     */
    inline static Vector3 random(double min, double max) {
        return Vector3(
                generateRandomDouble(min, max),
                generateRandomDouble(min, max),
                generateRandomDouble(min, max)
                );
    }

    /*
     * Check whether the given vector is close to zero vector
     */
    bool nearZero() const {
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

private:
    double e[3];
};

    /*
     * Type aliases for 'Vector3'
     */
    using Point3 = Vector3;   // 3D point
    using Color = Vector3;   // RGB Color

    std::istream& operator>>(std::istream &input_stream, Vector3 &t) {
        double x, y, z;
        input_stream >> x >> y >> z;
        t.setX(x);
        t.setY(y);
        t.setZ(z);
        return input_stream;
    }

    std::ostream& operator<<(std::ostream &output_stream, Vector3 &t) {
        output_stream << t.X() << " " << t.Y() << " " << t.Z();
        return output_stream;
    }

    /*
     * Overload binary operators for 'Vector3' class
     */
    Vector3 operator + (const Vector3 &v1, const Vector3 &v2) {
        return Vector3(v1.X() + v2.X(), v1.Y() + v2.Y(), v1.Z() + v2.Z());
    }

    Vector3 operator - (const Vector3 &v1, const Vector3 &v2) {
        return Vector3(v1.X() - v2.X(), v1.Y() - v2.Y(), v1.Z() - v2.Z());
    }

    Vector3 operator * (const Vector3 &v1, const Vector3 &v2) {
        return Vector3(v1.X() * v2.X(), v1.Y() * v2.Y(), v1.Z() * v2.Z());
    }

    Vector3 operator * (double t, const Vector3 &v) {
        return Vector3(t * v.X(), t * v.Y(), t * v.Z());
    }

    Vector3 operator * (const Vector3 &v, double t) {
        return t * v;
    }

    Vector3 operator / (Vector3 v, double t) {
        return (1/t) * v;
    }

    /**
     * Return normalized duplicate of 'v'
     * @param v a 'Vector3' instance
     * @return a duplicate of 'v' but normalized one
     */
    Vector3 normalize(Vector3 v) {
        return v / v.length();
    }

    /**
     * Calculate the value of inner product of two given vectors
     * @param v1 a 'Vector3' instance
     * @param v2 a 'Vector3' instance
     * @return the inner product of two vectors
     */
    double dotProduct(const Vector3 &v1, const Vector3 &v2) {
        return v1.X() * v2.X() + v1.Y() * v2.Y() + v1.Z() * v2.Z();
    }

    /**
     * Calculate the value of cross product of two given vectors
     * @param v1 a 'Vector3' instance
     * @param v2 a 'Vector3' instance
     * @return the cross product of two vectors
     */
    Vector3 crossProduct(const Vector3 &v1, const Vector3 &v2) {
        return Vector3((v1.Y() * v2.Z() - v1.Z() * v2.Y()),
                       (-(v1.X() * v2.Z() - v1.Z() * v2.X())),
                       (v1.X() * v2.Y() - v1.Y() * v2.X()));
    }

    /**
     * Calculate the reflection of 'v' which reflects off the surface characterized by HitPointNormal 'n'
     * @param v a 'Vector3' instance representing an incident Ray
     * @param n a 'Vector3' instance representing a HitPointNormal vector of the surface on which 'v' is being reflected off
     * @return the reflection of 'v'
     */
    Vector3 reflect(const Vector3& v, const Vector3& n) {
        return v - 2 * dotProduct(v, n) * n;
    }

    /**
     * Calculate the refraction of 'v' which refracts getPointAt the surface characterized by HitPointNormal 'n'
     * @param v a 'Vector3' instance representing an incident Ray
     * @param n a 'Vector3' instance representing a HitPointNormal vector of the surface on which 'v' is being refracted
     * @param etai_over_etat a real valued, ratio of refractive indices of two adjacent matters
     * @return the refraction of 'v'
     */
    Vector3 refract(const Vector3& v, const Vector3& n, double etai_over_etat) {
        auto cos_theta = fmin(dotProduct(-v, n), 1.0);
        Vector3 refracted_perp = etai_over_etat * (v + cos_theta * n);
        Vector3 refracted_parallel = -sqrt(fabs(1.0 - refracted_perp.lengthSquared())) * n;
        Vector3 refracted = refracted_perp + refracted_parallel;
        return refracted;
    }

    Vector3 randomUnitVector() {
        auto a = generateRandomDouble(0, 2 * Pi);
        auto z = generateRandomDouble(-1, 1);
        auto r = sqrt(1-z*z);
        return Vector3(r * cos(a), r * sin(a), z);
    }

    /**
     * Generate an unit vector which points an arbitrary point on the surface of unit Sphere
     * @return a normalized 'Vector3' instance that points a random point on a unit Sphere
     */
    Vector3 randomInUnitSphere() {
        while (true) {
            auto p = Vector3::random(-1, 1);
            if (p.lengthSquared() >= 1) continue;
            return normalize(p);
        }
    }

    /**
     * Generates an unit vector which points an arbitrary point of the unit disk
     * @return a normalized 'Vector3' instance that points a random point on a unit disk
     */
    Vector3 randomInUnitDisk() {
        while (true) {
            auto p = Vector3(generateRandomDouble(-1, 1), generateRandomDouble(-1, 1), 0.0);
            if (p.lengthSquared() >= 1) continue;
            return normalize(p);
        }
    }

    /**
     * Generate an unit vector which points an arbitrary point on the surface of hemisphere specified by HitPointNormal
     *
     * This function is used to generate a random Ray scattered on the surface of diffuse Material.
     *
     * By forcing a Ray to be emitted through a hemisphere in the getRayDirection of HitPointNormal vector,
     * one can be ensured that the object will behave more realistically under direct light source, since all
     * the rays are now spreads out uniformly to the open space.
     * @param NormalVector a HitPointNormal vector that points the center of unit Sphere tangent to the surface specified by itself
     * @return a randomly generated vector which points an arbitrary point on the hemisphere
     */
    Vector3 randomInHemisphere(const Vector3& NormalVector) {
        Vector3 in_unit_sphere = randomInUnitSphere();
        if (dotProduct(in_unit_sphere, NormalVector) > 0.0)
            // Ray vector points outward
            return in_unit_sphere;
        else
            // Ray vector points inward
            return -in_unit_sphere;

    }

#endif //RTFOUNDATION_VECTOR3_HPP
