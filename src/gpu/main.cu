//
// Created by dveloperY0115 on 1/8/2021.
//

#include <iostream>
#include "vector3.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hittable_list.hpp"

#define colorDim 3
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

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

__global__ void create_world(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vector3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vector3(0,-100.5,-1), 100);
        *d_world    = new hittable_list(d_list,2);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

__device__ vector3 ray_color(const ray& r, hittable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vector3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }

    vector3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render(float* fb, int max_x, int max_y, vector3 lower_left_corner, vector3 horizontal,
                       vector3 vertical, vector3 origin, hittable **world) {
    // get global pixel coordinate
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;   // don't render outside the image
    unsigned int pixel_index = y * max_x * colorDim + x * colorDim;
    float u = float(x) / float(max_x);
    float v = float(y) / float(max_y);

    ray r = ray(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = ray_color(r, world)[0];
    fb[pixel_index + 1] = ray_color(r, world)[1];
    fb[pixel_index + 2] = ray_color(r, world)[2];
}

int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Camera

    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vector3(viewport_width, 0, 0);
    auto vertical = vector3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vector3(0, 0, focal_length);

    // world

    hittable **d_list;
        checkCudaErrors(cudaMalloc((void**) &d_list, 2 * sizeof(hittable*)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hittable*)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int num_pixels = image_width * image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate Frame Buffer for rendering
    float *fb;
    checkCudaErrors(cudaMallocManaged((void**) &fb, fb_size));

    // set dimensions of grid
    int tx = 8;
    int ty = 8;

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, image_width, image_height, lower_left_corner, horizontal, vertical, origin, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width * colorDim + i * colorDim;
            float r = fb[pixel_index];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    free_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    return 0;
}