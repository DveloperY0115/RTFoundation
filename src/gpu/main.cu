//
// Created by dveloperY0115 on 1/8/2021.
//

#include <iostream>
#include <curand_kernel.h>
#include "vector3.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hittable_list.hpp"
#include "camera.hpp"

#define colorDim 3
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define RANDVEC3 vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
    vector3 p;
    do {
        p = 2.0f * RANDVEC3 - vector3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // reset device before terminating
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const point3& center, float radius, const ray& r) {
    vector3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    return (discriminant > 0);
}

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vector3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vector3(0,-100.5,-1), 100);
        *d_world    = new hittable_list(d_list,2);
        *d_camera = new camera();
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera **d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
    delete *d_camera;
}

__device__ vector3 ray_color(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    float cur_attenuation = 1.0f;

    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, target - rec.p);
        } else {
            vector3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vector3 color = (1.0f - t) * vector3(1.0, 1.0, 1.0) + t * vector3(0.5, 0.7, 1.0);
            return cur_attenuation * color;
        }
    }
    return vector3(0.0, 0.0, 0.0);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    unsigned int pixel_index = y * max_x + x;

    // each thread gets same seed, a different sequence number
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vector3* fb, int max_x, int max_y, int num_samples, camera** cam, hittable **world, curandState* rand_state) {
    // get global pixel coordinate
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;   // don't render outside the image
    unsigned int pixel_index = y * max_x + x;
    curandState local_rand_state = rand_state[pixel_index];
    vector3 pixel_color = color(0, 0, 0);
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(y + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        pixel_color += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    pixel_color /= float(num_samples);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);

    fb[pixel_index] = pixel_color;
}

int main() {

    // Image
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 1600;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    static const int num_samples = 50;

    // allocate Frame Buffer for rendering
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vector3);
    vector3 *fb;
    checkCudaErrors(cudaMallocManaged((void**) &fb, fb_size));

    // set random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**) &d_rand_state, num_pixels*sizeof(curandState)));

    // set world and camera
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void**) &d_list, 2 * sizeof(hittable*)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hittable*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // set dimensions of grid
    int tx = 8;
    int ty = 8;
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    // render
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, image_width, image_height, 50, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // write output to a file
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            float r = fb[pixel_index].r();
            float g = fb[pixel_index].g();
            float b = fb[pixel_index].b();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
    return 0;
}