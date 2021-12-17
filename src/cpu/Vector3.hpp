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

    Vector3(double element_0, double element_1, double element_2) {
        e[0] = element_0;
        e[1] = element_1;
        e[2] = element_2;
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

    double operator[](int index) const {
        return e[index];
    }

    double& operator[](int index) {
        return e[index];
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
        return sqrt(length_squared());
    }

    /*
     * Compute the squared sum of vector elements.
     */
    double length_squared() const {
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
                random_double(),
                random_double(),
                random_double()
                );
    }

    /*
     * Create a 3D vector whose elements are initialized with random numbers lying in the given interval.
     */
    inline static Vector3 random(double min, double max) {
        return Vector3(
                random_double(min, max),
                random_double(min, max),
                random_double(min, max)
                );
    }

    /*
     * Check whether the given vector is close to zero vector
     */
    bool near_zero() const {
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
    Vector3 unit_vector(Vector3 v) {
        return v / v.length();
    }

    /**
     * Calculate the value of inner product of two given vectors
     * @param v1 a 'Vector3' instance
     * @param v2 a 'Vector3' instance
     * @return the inner product of two vectors
     */
    double dot_product(const Vector3 &v1, const Vector3 &v2) {
        return v1.X() * v2.X() + v1.Y() * v2.Y() + v1.Z() * v2.Z();
    }

    /**
     * Calculate the value of cross product of two given vectors
     * @param v1 a 'Vector3' instance
     * @param v2 a 'Vector3' instance
     * @return the cross product of two vectors
     */
    Vector3 cross_product(const Vector3 &v1, const Vector3 &v2) {
        return Vector3((v1.Y() * v2.Z() - v1.Z() * v2.Y()),
                       (-(v1.X() * v2.Z() - v1.Z() * v2.X())),
                       (v1.X() * v2.Y() - v1.Y() * v2.X()));
    }

    /**
     * Calculate the reflection of 'v' which reflects off the surface characterized by normal 'n'
     * @param v a 'Vector3' instance representing an incident ray
     * @param n a 'Vector3' instance representing a normal vector of the surface on which 'v' is being reflected off
     * @return the reflection of 'v'
     */
    Vector3 reflect(const Vector3& v, const Vector3& n) {
        return v - 2 * dot_product(v, n) * n;
    }

    /**
     * Calculate the refraction of 'v' which refracts at the surface characterized by normal 'n'
     * @param v a 'Vector3' instance representing an incident ray
     * @param n a 'Vector3' instance representing a normal vector of the surface on which 'v' is being refracted
     * @param etai_over_etat a real valued, ratio of refractive indices of two adjacent matters
     * @return the refraction of 'v'
     */
    Vector3 refract(const Vector3& v, const Vector3& n, double etai_over_etat) {
        auto cos_theta = fmin(dot_product(-v, n), 1.0);
        Vector3 refracted_perp = etai_over_etat * (v + cos_theta * n);
        Vector3 refracted_parallel = -sqrt(fabs(1.0 - refracted_perp.length_squared())) * n;
        Vector3 refracted = refracted_perp + refracted_parallel;
        return refracted;
    }

    Vector3 random_unit_vector() {
        auto a = random_double(0, 2*pi);
        auto z = random_double(-1, 1);
        auto r = sqrt(1-z*z);
        return Vector3(r * cos(a), r * sin(a), z);
    }

    /**
     * Generate an unit vector which points an arbitrary point on the surface of unit sphere
     * @return a normalized 'Vector3' instance that points a random point on a unit sphere
     */
    Vector3 random_in_unit_sphere() {
        while (true) {
            auto p = Vector3::random(-1, 1);
            if (p.length_squared() >= 1) continue;
            return unit_vector(p);
        }
    }

    /**
     * Generates an unit vector which points an arbitrary point of the unit disk
     * @return a normalized 'Vector3' instance that points a random point on a unit disk
     */
    Vector3 random_in_unit_disk() {
        while (true) {
            auto p = Vector3(random_double(-1, 1), random_double(-1, 1), 0.0);
            if (p.length_squared() >= 1) continue;
            return unit_vector(p);
        }
    }

    /**
     * Generate an unit vector which points an arbitrary point on the surface of hemisphere specified by normal
     *
     * This function is used to generate a random ray scattered on the surface of diffuse material.
     *
     * By forcing a ray to be emitted through a hemisphere in the direction of normal vector,
     * one can be ensured that the object will behave more realistically under direct light source, since all
     * the rays are now spreads out uniformly to the open space.
     * @param normal a normal vector that points the center of unit sphere tangent to the surface specified by itself
     * @return a randomly generated vector which points an arbitrary point on the hemisphere
     */
    Vector3 random_in_hemisphere(const Vector3& normal) {
        Vector3 in_unit_sphere = random_in_unit_sphere();
        if (dot_product(in_unit_sphere, normal) > 0.0)
            // ray vector points outward
            return in_unit_sphere;
        else
            // ray vector points inward
            return -in_unit_sphere;

    }

#endif //RTFOUNDATION_VECTOR3_HPP
