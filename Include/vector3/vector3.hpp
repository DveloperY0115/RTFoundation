//
// Created by 유승우 on 2020/04/21.
//

#ifndef FIRSTRAYTRACER_VECTOR3_HPP
#define FIRSTRAYTRACER_VECTOR3_HPP

#include <math.h>
#include <stdlib.h>
#include <iostream>

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
     *  Defining basic arithmetic for a vector
     */
    vector3 operator - () const { return vector3(-e[0], -e[1], -e[2]); }
    double operator[](int index) const { return e[index]; }
    // Overriding?
    double& operator[](int index) { return e[index]; };

    /*
     *  Element-wise addition, subtraction 은 가능한데, 곱셈, 나눗셈? broadcasting 인가?
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

    /*
     *  Returns the Euclidean Norm and its square.
     */
    double length() const
    {
        return sqrt(length_squared());
    }

    double length_squared() const
    {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    // Creates an unit vector
    void make_unit_vector()
    {
        double k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] * e[2]*e[2]);
        e[0] *= k; e[1] *= k; e[2] *= k;
    }

    static vector3 random()
    {
        return vector3(random_double(), random_double(), random_double());
    }

    static vector3 random(double min, double max)
    {
        return vector3(random_double(min,max), random_double(min, max),
                random_double(min, max));
    }

    friend std::istream& operator>>(std::istream &input_stream, vector3 &t);
    friend std::ostream& operator<<(std::ostream &output_stream, vector3 &t);

};

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

    using point3 = vector3;   // 3D point
    using color = vector3;   // RGB color

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

    vector3 random_in_unit_sphere()
    {
        while (true)
        {
            auto p = vector3::random(-1, 1);
            if (p.length_squared() >= 1) continue;
            return p;
        }
    }



#endif //FIRSTRAYTRACER_VECTOR3_HPP
