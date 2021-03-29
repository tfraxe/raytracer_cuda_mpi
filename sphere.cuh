#ifndef SPHERE_H
#define SPHERE_H

#include <cmath>
#include "hittable.cuh"
#include "vec_3d.cuh"

class sphere : public hittable {
    public:
        __device__ sphere() {}
        __device__ sphere(point_3d c, float r, material* m) : mat_ptr(m), center(c), radius(r) {};

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    public:
        point_3d center;
        float radius;
        material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    vec_3d oc = r.orig() - this->center;
    float a = r.dir().length_squared();
    float half_b = oc.dot(r.dir()); 
    float c = oc.length_squared() - this->radius * this->radius;
    float delta = half_b * half_b - a * c;
    if (delta < 0) return false;

    float sqrt_delta = std::sqrt(delta);
    float t_root = (-half_b - sqrt_delta)/a;
    if (t_root < t_min || t_root > t_max) {
        t_root = (-half_b + sqrt_delta)/a;
            if(t_root < t_min || t_root > t_max) return false;
    }

    rec.t = t_root;
    rec.hit_point = r.point_at(t_root);
    vec_3d normal = (rec.hit_point - this->center) / this->radius;
    rec.set_face_normal(r, normal);
    rec.mat_ptr = this->mat_ptr;
    return true;
}


#endif
