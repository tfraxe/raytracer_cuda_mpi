#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec_3d.cuh"
#include "ray.cuh"
#include "hittable.cuh"

class material {
    public:
        __device__ virtual bool scatter(const ray& r_incident, const hit_record& record, vec_3d& attenuation, 
                ray& scattered, curandState* local_rand_state) const = 0;
};


class lambertian : public material {
    public:
        __device__ lambertian(const vec_3d& a) : albedo(a) {}

        __device__ bool scatter(const ray& r_incident, const hit_record& record, 
                vec_3d& attenuation, ray& scattered, curandState* local_rand_state) const override {

            vec_3d scatter_direction = record.normal + random_in_unit_sphere(local_rand_state).unit_vector();

            if (scatter_direction.near_zero()) scatter_direction = record.normal;

            scattered = ray(record.hit_point, scatter_direction);
            attenuation = albedo;
            return true;
        }

    public:
        vec_3d albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec_3d& a, float f) : albedo(a), fuzz( f < 1 ? f : f) {}

        __device__ virtual bool scatter(const ray& r_incident, const hit_record& record, vec_3d& attenuation, 
                ray& scattered, curandState* local_rand_state) const override {

            vec_3d reflected_ray = reflect(r_incident.dir().unit_vector(), record.normal);

            scattered = ray(record.hit_point, reflected_ray + this->fuzz * random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return scattered.dir().dot(record.normal) > 0.0f;
        }

    public:
        vec_3d albedo;
        float fuzz;
};

class dielectric : public material {
    public:
       __device__ dielectric(float ri) : refraction_index(ri) {}

       __device__ virtual bool scatter(const ray& r_incident, const hit_record& record, vec_3d& attenuation, 
                ray& scattered, curandState* local_rand_state) const override {

           attenuation = vec_3d(1.0f, 1.0f, 1.0f);
           float refraction_ratio = record.front_face ? (1.0f/refraction_index) : refraction_index;
           vec_3d unit_direction = r_incident.dir().unit_vector();
           
           float cos_theta = fmin(-unit_direction.dot(record.normal), 1.0f);
           float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

           vec_3d scattered_direction;
           if (refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)) {
               scattered_direction = reflect(unit_direction, record.normal); 
           } else {
               scattered_direction = refract(unit_direction, record.normal, refraction_ratio);
           }


           scattered = ray(record.hit_point, scattered_direction);
           return true;

       }

    public:
       float refraction_index;

    private:
       __device__ static float reflectance(const float cosine, const float ref_idx) {
            float r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1-r0)*pow(1 - cosine, 5);
       }
};



#endif
