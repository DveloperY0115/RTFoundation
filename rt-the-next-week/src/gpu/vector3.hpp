#ifndef VECTOR3_CUDAH
#define VECTOR3_CUDAH

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <curand_kernel.h>

class vector3  {

public:
    __host__ __device__ vector3() : e{0, 0, 0}
    {
        // Do nothing
    }

    __host__ __device__ vector3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vector3& operator+() const { return *this; }
    __host__ __device__ inline vector3 operator-() const { return vector3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vector3& operator+=(const vector3 &v2);
    __host__ __device__ inline vector3& operator-=(const vector3 &v2);
    __host__ __device__ inline vector3& operator*=(const vector3 &v2);
    __host__ __device__ inline vector3& operator/=(const vector3 &v2);
    __host__ __device__ inline vector3& operator*=(const float t);
    __host__ __device__ inline vector3& operator/=(const float t);

    __host__ __device__ inline double length() const { return (double)sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline double squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();


    float e[3];
};

using point3 = vector3;
using color = vector3;

inline std::istream& operator>>(std::istream &is, vector3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vector3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void vector3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vector3 operator+(const vector3 &v1, const vector3 &v2) {
    return vector3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vector3 operator-(const vector3 &v1, const vector3 &v2) {
    return vector3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vector3 operator*(const vector3 &v1, const vector3 &v2) {
    return vector3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vector3 operator/(const vector3 &v1, const vector3 &v2) {
    return vector3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vector3 operator*(float t, const vector3 &v) {
    return vector3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vector3 operator/(vector3 v, float t) {
    return vector3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vector3 operator*(const vector3 &v, float t) {
    return vector3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline double dot(const vector3 &v1, const vector3 &v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] *v2.e[1] + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vector3 cross(const vector3 &v1, const vector3 &v2) {
    return vector3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                 (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                 (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


__host__ __device__ inline vector3& vector3::operator+=(const vector3 &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline vector3& vector3::operator*=(const vector3 &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline vector3& vector3::operator/=(const vector3 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vector3& vector3::operator-=(const vector3& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline vector3& vector3::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline vector3& vector3::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vector3 unit_vector(vector3 v) {
    return v / v.length();
}

#define RANDVEC3 vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
    vector3 p;
    do {
        p = 2.0f * RANDVEC3 - vector3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vector3 random_in_unit_disk(curandState *local_rand_state) {
    while (true) {
        auto p = vector3(random_double(local_rand_state), random_double(local_rand_state), 0.0);
        if (p.squared_length() >= 1) continue;
        return unit_vector(p);
    }
}

__device__ vector3 reflect(const vector3& v, const vector3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ vector3 refract(const vector3& v, const vector3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-v, n), 1.0);
    vector3 refracted_perp = etai_over_etat * (v + cos_theta * n);
    vector3 refracted_parallel = -sqrt(fabs(1.0 - refracted_perp.squared_length())) * n;
    vector3 refracted = refracted_perp + refracted_parallel;
    return refracted;
}

#endif