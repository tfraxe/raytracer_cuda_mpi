#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>
#include "vec_3d.cuh"
#include "ray.cuh"

#include "utility.hpp"

class camera {
    public:
        __device__ __host__ camera(
                const point_3d& lookfrom,
                const point_3d& lookat,
                const vec_3d& vup,
                const float vertical_fov, 
                const float aspect_ratio,
                const float aperture,
                const float focus_dist,
                const int p_rank,
                const int number_of_divisions
            ) {
           
            float theta = degrees_to_radians(vertical_fov);
            const float h = tan(theta/2);
            const float viewport_height = 2.0f * h / number_of_divisions;
            const float viewport_width = number_of_divisions * viewport_height * aspect_ratio;

            w = (lookfrom - lookat).unit_vector();
            u = vup.cross(w).unit_vector();
            v = w.cross(u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin + ((float(p_rank) - 1.0f - float(number_of_divisions)/2.0f) * vertical) - horizontal/2 - focus_dist * w;

            lens_radius = aperture / 2;

        }

        __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
            vec_3d rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec_3d offset = rd.x() * u + rd.y() * v;

            return ray(origin + offset, 
                    lower_left_corner +  s * horizontal + t * vertical - origin - offset);
        }

        


    public:
        point_3d origin;
        vec_3d horizontal;
        vec_3d vertical;
        point_3d lower_left_corner;
        vec_3d u, v, w;
        float lens_radius;
        

};

#endif
