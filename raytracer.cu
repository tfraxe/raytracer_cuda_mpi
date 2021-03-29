#include <iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include <cmath>

#include "vec_3d.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "hittable_list.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "utility.hpp"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}


__device__ float did_hit_sphere(const point_3d sphere_center, float sphere_radius, const ray r) {
    vec_3d oc = r.orig() - sphere_center;
    float a = r.dir().length_squared();
    float half_b = oc.dot(r.dir()); 
    float c = oc.length_squared() - sphere_radius * sphere_radius;
    float delta = half_b * half_b - a * c;
    if (delta < 0) return -1;
    else return (- half_b - sqrt(delta))/a;
}

__device__ vec_3d ray_color(const ray& r, hittable** d_world, curandState* local_rand_state) {

    ray current_ray = r;
    vec_3d current_attenuation(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 50; i++) {
        hit_record record;
        if ((*d_world)->hit(current_ray, 0.001f, infinity, record)) {
            ray scattered;
            vec_3d attenuation;

            if(record.mat_ptr->scatter(current_ray, record, attenuation, scattered, local_rand_state)) {
                current_ray = scattered; 
                current_attenuation *= attenuation;
            } else {
                return vec_3d(0, 0, 0);
            }

        } else {
            const vec_3d unit_vector = current_ray.dir().unit_vector();
            float s = 0.5f * (unit_vector.y() + 1.0f);
            vec_3d final_color =  (1.0f - s) * vec_3d(1.0f, 1.0f, 1.0f) + s * vec_3d(0.5f, 0.7f, 1.0f);
            return current_attenuation * final_color;
        }
    }

    return vec_3d(0, 0, 0);
}
__global__
void rand_init(curandState* rand_state) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__
void render_init_rand(const int height, const int width, curandState* rand_state) {
   int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
   int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((thread_x >= width) || (thread_y >= height)) return;

   int current_index = thread_y * width + thread_x;
   curand_init(1984 + current_index, 0, 0, &rand_state[current_index]);

}

__global__
void render(hittable** d_world, const int height, const int width, float* frame_buffer, const int number_of_samples, curandState* rand_state, camera** cam) {

   int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
   int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((thread_x >= width) || (thread_y >= height)) return;

   float v = float(thread_y)/(height);
   float u = float(thread_x)/(width);
   int current_index = thread_y * width + thread_x;
   curandState local_rand_state = rand_state[current_index];

   vec_3d color(0, 0, 0);
   for (int i = 0; i < number_of_samples; i++) {
       ray r = (*cam)->get_ray(
               u + curand_uniform(&local_rand_state)/float(width),
               v + curand_uniform(&local_rand_state)/float(height), 
               &local_rand_state);

       color += ray_color(r, d_world, &local_rand_state);
   }

   frame_buffer[3 * current_index + 0] = color.data[0];
   frame_buffer[3 * current_index + 1] = color.data[1];
   frame_buffer[3 * current_index + 2] = color.data[2];
   
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_scene(hittable** d_objects, hittable** d_world, camera** d_cam, float aspect_ratio, int p_rank, int number_of_divisions, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;
        d_objects[0] = new sphere(vec_3d(0, -1000.0f, -1), 1000, new lambertian(vec_3d(0.5, 0.5, 0.5)));

        int i = 1;

        for (int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec_3d center(a+0.9*RND, 0.2, b+0.9*RND);

                if(choose_mat < 0.8f) d_objects[i++] = new sphere(center, 0.2, new lambertian(vec_3d(RND*RND, RND*RND, RND*RND)));
                else if (choose_mat < 0.95f) d_objects[i++] = new sphere(center, 0.2, new metal(vec_3d(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f*RND));
                else d_objects[i++] = new sphere(center, 0.2, new dielectric(1.5));
            }

        }

        d_objects[i++] = new sphere(vec_3d(0, 1, 0), 1.0f, new dielectric(1.5));
        d_objects[i++] = new sphere(vec_3d(-4, 1, 0), 1.0f, new lambertian(vec_3d(0.4, 0.2, 0.1)));
        d_objects[i++] = new sphere(vec_3d(4, 1, 0), 1.0f, new metal(vec_3d(0.7f, 0.6f, 0.5f), 0.0f));
        *rand_state = local_rand_state;

        *d_world = new hittable_list(d_objects, 1 + 22*22 + 3);

        point_3d lookfrom(13, 2, 3);
        point_3d lookat(0, 0, 0);
        vec_3d vup(0, 1, 0);
        float dist_to_focus = 10.0f;
        float aperture = 0.1f;

        *d_cam = new camera(lookfrom, lookat, vup, 30, aspect_ratio, aperture, dist_to_focus, p_rank, number_of_divisions);

    }
}

__global__ void cleanup_scene(hittable** d_objects, hittable** d_world, camera** d_cam) {
    for (int i = 0; i < 1 + 22*22 + 3; i++) {
        delete ((sphere*) d_objects[i])->mat_ptr;
        delete d_objects[i];
    }
    delete *d_world;
    delete *d_cam;
}

void run_raytracer(int p_rank, int number_of_divisions, float* out_fb, int image_width, float aspect_ratio, const float number_of_samples) {

    const int image_height = static_cast<int>((image_width / aspect_ratio) * 1/number_of_divisions);

    
    hittable **d_objects, **d_world;
    camera** d_cam;
    checkCudaErrors(cudaMalloc(&d_objects, (22*22+4)*sizeof(hittable *)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable *)));
    checkCudaErrors(cudaMalloc(&d_cam, sizeof(camera *)));
        

    float* frame_buffer;
    checkCudaErrors(cudaMallocManaged(&frame_buffer, 3*image_height*image_width*sizeof(float)));

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, image_height*image_width*sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc(&d_rand_state2, 1*sizeof(curandState)));

    cudaStream_t cuda0;
    cudaStreamCreate(&cuda0);
    std::cerr << "stream: " << cuda0 << '\n';

    rand_init<<<1, 1, 0, cuda0>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(cuda0));


    create_scene<<<1, 1, 0, cuda0>>>(d_objects, d_world, d_cam, aspect_ratio, p_rank, number_of_divisions, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(cuda0));

    int threadsDim = 32;
    dim3 numThreads(threadsDim, threadsDim);
    dim3 numBlocks((image_width / threadsDim) + 1, (image_height / threadsDim) + 1);

    render_init_rand<<<numBlocks, numThreads, 0, cuda0>>>(image_height, image_width, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(cuda0));

    
    render<<<numBlocks,numThreads, 0, cuda0>>>(d_world, image_height, image_width, frame_buffer, number_of_samples, d_rand_state, d_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(cuda0));

    cudaMemcpyAsync(out_fb, frame_buffer, 3*image_height*image_width * sizeof(float), cudaMemcpyDeviceToHost, cuda0);

    checkCudaErrors(cudaStreamSynchronize(cuda0));
    cleanup_scene<<<1, 1, 0, cuda0>>>(d_objects, d_world, d_cam);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceReset();

}
