#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cuh"

class hittable_list : public hittable {
    public:
        __device__ hittable_list() {}
        __device__ hittable_list(hittable** l, int s) : list(l), size(s) {}
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override;
    public:
        hittable** list;
        int size;
};

__device__ bool hittable_list::hit(const ray&r, float tmin, float tmax, hit_record& rec) const {
    hit_record temp_record;
    bool did_hit = false;
    float closest_t = tmax;

    for (int i = 0; i < this->size; i++) {
        if(list[i]->hit(r, tmin, closest_t, temp_record)) {
            did_hit = true;
            closest_t = temp_record.t;
            rec = temp_record;
        }
    }

    return did_hit;
}


#endif
