//
// Created by 유승우 on 2020/04/21.
//
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "vector3/vector3.h"

    /*
     *  Constructor for 3D vector
     */
    vector3::vector3(double element_0, double element_1, double element_2) { e[0] = element_0; e[1] = element_1; e[2] = element_2; }
    /*
     *  Getters for position, rgb values of each pixel.
     */
    double vector3::getX() const { return e[0]; }
    double vector3::getY() const { return e[1]; }
    double vector3::getZ() const { return e[2]; }

    /*
     *  Defining basic arithmetic for a vector
     */
    vector3 vector3::operator - () const { return vector3(-e[0], -e[1], -e[2]); }
    double vector3::operator[](int index) const { return e[index]; }
    // Overriding?
    double& vector3::operator[](int index) { return e[index]; };

    /*
     *  Element-wise addition, subtraction 은 가능한데, 곱셈, 나눗셈? broadcasting 인가?
     */
    vector3& vector3::operator += (const vector3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vector3& vector3::operator -= (const vector3 &v)
    {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    vector3& vector3::operator *= (const vector3 &v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    vector3& vector3::operator /= (const vector3 &v)
    {
        e[0] /= v.e[0];
        e[1] /= v.e[1];
        e[2] /= v.e[2];
        return *this;
    }

    vector3& vector3::operator *= (const double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vector3& vector3::operator /= (const double t)
    {
        return *this *= 1/t;
    }

    /*
    *  Returns the Euclidean Norm and its square.
    */
    double vector3::length() const
    {
        return sqrt(length_squared());
    }

    double vector3::length_squared() const
    {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    void vector3::make_unit_vector()
    {
        double k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] * e[2]*e[2]);
        e[0] *= k; e[1] *= k; e[2] *= k;
    }

    void vector3::write_color(std::ostream &out)
    {
        // Write the translated [0, 255] value of each color component.
        out << static_cast<int>(255.999 * e[0]) << ' '
            << static_cast<int>(255.999 * e[1]) << ' '
            << static_cast<int>(255.999 * e[2]) << '\n';
    }

    // vector3 Utility Functions

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

    vector3 unit_vector(vector3 v)
    {
        return v / v.length();
    }

    double dot_product(const vector3 &v1, const vector3 &v2)
    {
        return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
    }

    vector3 cross_product(const vector3 &v1, const vector3 &v2)
    {
        return vector3( (v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
    }

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