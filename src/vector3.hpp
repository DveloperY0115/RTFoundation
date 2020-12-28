//
// Created by 유승우 on 2020/04/21.
//

#ifndef FIRSTRAYTRACER_VECTOR3_HPP
#define FIRSTRAYTRACER_VECTOR3_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

class vector3
{
public:
    double e[3]{};
    /*
     *  Empty constructor for 3D vector
     */
    vector3() : e{0, 0, 0}
    {
        // Do nothing
    }

    vector3(double element_0, double element_1, double element_2)
    { e[0] = element_0; e[1] = element_1; e[2] = element_2; }

    /*
     *  Getters for position, rgb values of each pixel.
     */
    double getX() const { return e[0]; }
    double getY() const { return e[1]; }
    double getZ() const { return e[2]; }

    /*
     *  Define unary operators for a vector
     */
    vector3 operator - () const { return vector3(-e[0], -e[1], -e[2]); }

    double operator[](int index) const { return e[index]; }
    // Overriding?
    double& operator[](int index) { return e[index]; };

    /*
     * Define vector addition, subtraction, (element-wise) multiplication and division,
     * and scalar multiplication and division
     */
    vector3& operator += (const vector3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vector3& operator -= (const vector3 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    vector3& operator *= (const vector3 &v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    vector3& operator /= (const vector3 &v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        return *this;
    }

    vector3& operator *= (const double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vector3& operator /= (const double t)
    {
        return *this *= 1/t;
    }

    /*
     * 왜 binary addition operator는 클래스 내에서 overloading이 안 되지?
     * 매개 변수로 클래스의 두 인스턴스를 받기 때문에 메서드로 선언하면 다른 게 뭔지 모름!
     * vector3 operator + (const vector3 &v1, const vector3 &v2);
     */

    /**
     * Return the Euclidean norm of calling vector
     * @return Euclidean length of vector
     */
    double length() const
    {
        return sqrt(length_squared());
    }

    /**
     * Return the squared sum of vector elements (square of length)
     * @return Square of Euclidean length of vector
     */
    double length_squared() const
    {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    /**
     * Normalize the calling vector
     */
    void make_unit_vector()
    {
        double k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] * e[2]*e[2]);
        e[0] *= k; e[1] *= k; e[2] *= k;
    }

    /**
     * Create and return an instance of 'vector3' class
     * @return an instance of 'vector3' whose elements are set randomly
     */
    inline static vector3 random()
    {
        return vector3(random_double(), random_double(), random_double());
    }

    /**
     * Create and return an instance of 'vector3' class
     * @return an instance of 'vector3' whose elements are set randomly within given bounds
     */
    inline static vector3 random(double min, double max)
    {
        return vector3(random_double(min,max), random_double(min, max),
                random_double(min, max));
    }

    friend std::istream& operator>>(std::istream &input_stream, vector3 &t);
    friend std::ostream& operator<<(std::ostream &output_stream, vector3 &t);

};

    /*
     * Type aliases for 'vector3'
     */
    using point3 = vector3;   // 3D point
    using color = vector3;   // RGB color

    std::istream& operator>>(std::istream &input_stream, vector3 &t)
    {
        input_stream >> t.e[0] >> t.e[1] >> t.e[2];
        return input_stream;
    }

    std::ostream& operator<<(std::ostream &output_stream, vector3 &t)
    {
        output_stream << t.e[0] << " " << t.e[1] << " " << t.e[2];
        return output_stream;
    }

    /*
     * Overload binary operators for 'vector3' class
     */
    vector3 operator + (const vector3 &v1, const vector3 &v2)
    {
        return vector3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
    }

    vector3 operator - (const vector3 &v1, const vector3 &v2)
    {
        return vector3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
    }

    vector3 operator * (const vector3 &v1, const vector3 &v2)
    {
        return vector3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
    }

    vector3 operator * (double t, const vector3 &v)
    {
        return vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    vector3 operator * (const vector3 &v, double t)
    {
        return t * v;
    }

    vector3 operator / (vector3 v, double t)
    {
        return (1/t) * v;
    }

    /**
     * Return normalized duplicate of 'v'
     * @param v a 'vector3' instance
     * @return a duplicate of 'v' but normalized one
     */
    vector3 unit_vector(vector3 v)
    {
        return v / v.length();
    }

    /**
     * Calculate the value of inner product of two given vectors
     * @param v1 a 'vector3' instance
     * @param v2 a 'vector3' instance
     * @return the inner product of two vectors
     */
    double dot_product(const vector3 &v1, const vector3 &v2)
    {
        return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
    }

    /**
     * Calculate the value of cross product of two given vectors
     * @param v1 a 'vector3' instance
     * @param v2 a 'vector3' instance
     * @return the cross product of two vectors
     */
    vector3 cross_product(const vector3 &v1, const vector3 &v2)
    {
        return vector3( (v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
    }

    /**
     * Calculate the reflection of 'v' which reflects off the surface characterized by normal 'n'
     * @param v a 'vector3' instance representing an incidence ray
     * @param n a 'vector3' instance representing a normal vector of the surface on which 'v' is being reflected off
     * @return the reflection of 'v'
     */
    vector3 reflect(const vector3& v, const vector3& n)
    {
        return v - 2 * dot_product(v, n) * n;
    }

    vector3 random_unit_vector()
    {
        auto a = random_double(0, 2*pi);
        auto z = random_double(-1, 1);
        auto r = sqrt(1-z*z);
        return vector3(r*cos(a), r*sin(a), z);
    }

    /**
     * Generate an unit vector which points an arbitrary point on the surface of unit sphere
     * @return a normalized 'vector3' instance that points a random point on a unit sphere
     */
    vector3 random_in_unit_sphere()
    {
        while (true)
        {
            auto p = vector3::random(-1, 1);
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
    vector3 random_in_hemisphere(const vector3& normal)
    {
        vector3 in_unit_sphere = random_in_unit_sphere();
        if (dot_product(in_unit_sphere, normal) > 0.0)
            return in_unit_sphere;
        else
            return -in_unit_sphere;

    }



#endif //FIRSTRAYTRACER_VECTOR3_HPP
