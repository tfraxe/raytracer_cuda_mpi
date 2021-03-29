#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.cuh"

class material;

struct hit_record {
    point_3d hit_point;
    vec_3d normal;
    material* mat_ptr;
    float t;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const vec_3d& outward_normal) {
        front_face = r.dir().dot(outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};


#endif
