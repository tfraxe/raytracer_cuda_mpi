#ifndef RAY_H
#define RAY_H

#include "vec_3d.cuh"

class ray {
    public: 
        point_3d origin;
        vec_3d direction;
    public:
        __device__ ray() {}
        __device__ ray(const point_3d& orig, const vec_3d& dir) : origin(orig), direction(dir) {}
        __device__ point_3d orig() const { return origin; }
        __device__ vec_3d dir() const { return direction; }

        __device__ point_3d point_at(float t) const { return origin + t*direction; }


};


#endif
