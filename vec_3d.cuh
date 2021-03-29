#ifndef VEC_3D_H
#define VEC_3D_H

#include <iostream>
#include <cmath>
#include "utility.hpp"
#include <curand_kernel.h>

class vec_3d {
    public: 
        __host__ __device__ vec_3d() : data{0, 0, 0} {}
        __host__ __device__ vec_3d(float x, float y, float z) : data{x, y, z} {}

        __host__ __device__ float x() const { return data[0]; }
        __host__ __device__ float y() const { return data[1]; }
        __host__ __device__ float z() const { return data[2]; }
        __host__ __device__ float red() const { return data[0]; }
        __host__ __device__ float green() const { return data[1]; }
        __host__ __device__ float blue() const { return data[2]; }

        __host__ __device__ float operator[](int i) const { return data[i]; }
        __host__ __device__ float& operator[](int i) { return data[i]; }
        __host__ __device__ inline vec_3d operator+=(const vec_3d& v) {
            this->data[0] += v.data[0];
            this->data[1] += v.data[1];
            this->data[2] += v.data[2];

            return *this;
        }

        __host__ __device__ inline vec_3d operator*=(const vec_3d& v) {
            this->data[0] *= v.data[0];
            this->data[1] *= v.data[1];
            this->data[2] *= v.data[2];

            return *this;
        }

        __host__ __device__ float length_squared() { return data[0]*data[0] + data[1]*data[1] + data[2]*data[2]; }
        __host__ __device__ float length() { return sqrt(this->length_squared()); }
        __host__ __device__ vec_3d unit_vector();

        __host__ __device__ float dot(const vec_3d& v) const { return this->x() * v.x() + this->y() * v.y() + this->z() * v.z(); }

        __host__ __device__ vec_3d cross(const vec_3d& v) const {
                return vec_3d( 
                        (data[1] * v.data[2] - data[2] * v.data[1]),
                       -(data[0] * v.data[2] - data[2] * v.data[0]),
                        (data[0] * v.data[1] - data[1] * v.data[0])
                );
                                
        }

        __device__ bool near_zero() {
            const float s = 1e-8;
            return (fabs(this->data[0]) < s) && (fabs(this->data[1]) < s) && (fabs(this->data[2]) < s);
        }

        

        __host__ void write_as_color(std::ostream& out, const float number_of_samples) const {

            const float scale = 1/number_of_samples;
            const float red = sqrt(scale * this->red());
            const float green = sqrt(scale * this->green());
            const float blue = sqrt(scale * this->blue());

            out << static_cast<int>(256 * clamp(red, 0.0f, 0.999f)) << ' '
                << static_cast<int>(256 * clamp(green, 0.0f, 0.999f)) << ' '
                << static_cast<int>(256 * clamp(blue, 0.0f, 0.999f)) << '\n';
        }

    public:
        float data[3];
};

__host__ __device__ inline vec_3d operator-(const vec_3d& v) {
    return vec_3d(-v.x(), -v.y(), -v.z());
}
__host__ __device__ inline vec_3d operator+(const vec_3d& left, const vec_3d& right) { 
    return vec_3d(left.x() + right.x(), left.y() + right.y(), left.z() + right.z()); 
}

__host__ __device__ inline vec_3d operator*(const float& s, const vec_3d& vec) {
    return vec_3d(s * vec.x(), s * vec.y(), s * vec.z());
}

__host__ __device__ inline vec_3d operator*(const vec_3d& v1, const vec_3d& v2) {
    return vec_3d(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}

__host__ __device__ inline vec_3d operator/(const vec_3d& vec, const float& s) {
    return vec_3d(vec.x() / s, vec.y() / s, vec.z() / s);
}

__host__ __device__ inline vec_3d operator-(const vec_3d& vec, const vec_3d& vec2) {
    return vec_3d(vec.x() - vec2.x(), vec.y() - vec2.y(), vec.z() - vec2.z());
}

__host__ __device__ vec_3d vec_3d::unit_vector() { return *this / this->length(); }

__host__ inline std::ostream& operator<<(std::ostream &out, const vec_3d& v) {
    return out << v.data[0] << ' ' << v.data[1] << ' ' << v.data[2] << ' ';
}

__device__ inline vec_3d random_vector(curandState* local_rand_state) {
    return vec_3d(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state));
}

__device__ inline vec_3d random_in_unit_sphere(curandState* local_rand_state) {
    vec_3d p;
    while(true) {
        p = 2*random_vector(local_rand_state) - vec_3d(1, 1, 1);
        if(p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ inline vec_3d random_in_unit_disk(curandState* local_rand_state) {
    vec_3d p;
    while(true) {
        p = vec_3d(2 * curand_uniform(local_rand_state) - 1, 
                   2 * curand_uniform(local_rand_state) - 1,
                   0
                  );
        if(p.length_squared() >= 1.0f) continue;
        return p;
    }
}



__device__ vec_3d reflect(const vec_3d& v_in, const vec_3d& mirror_normal) {
            return v_in - 2 * v_in.dot(mirror_normal) * mirror_normal;
}

__device__ vec_3d refract(const vec_3d& uv, const vec_3d& normal, float etai_over_etat) {
    float cos_theta = fmin(-uv.dot(normal), 1.0f);

    vec_3d perp_component = etai_over_etat * (uv + cos_theta * normal);
    vec_3d parallel_component = -sqrt(fabs(1.0f - perp_component.length_squared())) * normal;

    return perp_component + parallel_component;
}


using point_3d = vec_3d;

#endif
