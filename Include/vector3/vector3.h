//
// Created by 유승우 on 2020/04/21.
//

#ifndef FIRSTRAYTRACER_VECTOR3_H
#define FIRSTRAYTRACER_VECTOR3_H

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
    vector3() : e{0, 0, 0} {}
    vector3(double element_0, double element_1, double element_2);
    /*
     *  Getters for position, rgb values of each pixel.
     */
    double getX() const;
    double getY() const;
    double getZ() const;

    /*
     *  Defining basic arithmetic for a vector
     */
    vector3 operator - () const;
    double operator[](int index) const;
    // Overriding?
    double& operator[](int index);

    /*
     *  Element-wise addition, subtraction 은 가능한데, 곱셈, 나눗셈? broadcasting 인가?
     */
    vector3& operator += (const vector3 &v);
    vector3& operator -= (const vector3 &v);
    vector3& operator *= (const vector3 &v);
    vector3& operator /= (const vector3 &v);
    vector3& operator *= (const double t);
    vector3& operator /= (const double t);

    /*
     * 왜 binary addition operator는 클래스 내에서 overloading이 안 되지?
     * 매개 변수로 클래스의 두 인스턴스를 받기 때문에 메서드로 선언하면 다른 게 뭔지 모름!
     * vector3 operator + (const vector3 &v1, const vector3 &v2);
     */

    /*
     *  Returns the Euclidean Norm and its square.
     */
    double length() const;
    double length_squared() const;

    void write_color(std::ostream &out);

    // Creates an unit vector
    void make_unit_vector();

    friend std::istream& operator>>(std::istream &input_stream, vector3 &t);
    friend std::ostream& operator<<(std::ostream &output_stream, vector3 &t);

};

using point3 = vector3;   // 3D point
using color = vector3;   // RGB color

    vector3 operator + (const vector3 &v1, const vector3 &v2);
    vector3 operator - (const vector3 &v1, const vector3 &v2);
    vector3 operator * (const vector3 &v1, const vector3 &v2);

    vector3 operator * (double t, const vector3 &v);
    vector3 operator * (const vector3 &v, double t);
    vector3 operator / (vector3 v, double t);
    vector3 unit_vector(vector3 v);

    double dot_product(const vector3 &v1, const vector3 &v2);
    vector3 cross_product(const vector3 &v1, const vector3 &v2);




#endif //FIRSTRAYTRACER_VECTOR3_H
