//
// Created by dveloperY0115 on 1/8/2021.
//

#include "rtweekend-gpu.hpp"
#include "vector3.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hittable_list.hpp"
#include "camera.hpp"

#include "material.hpp"
#include "lambertian.hpp"
#include "metal.hpp"
#include "dielectric.hpp"

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

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vector3(0,0,-1), 0.5,
                               new lambertian(color(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vector3(0,-100.5,-1), 100,
                               new lambertian(color(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vector3(1,0,-1), 0.5,
                               new metal(color(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vector3(-1,0,-1), 0.5,
                               new dielectric(1.5));
        d_list[4] = new sphere(vector3(-1,0,-1), -0.45,
                               new dielectric(1.5));
        *d_world  = new hittable_list(d_list,5);

        // initialize camera
        vector3 lookfrom(3, 3, 2);
        vector3 lookat(0, 0, -1);

        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 2.0f;
        *d_camera = new camera(
                lookfrom,
                lookat,
                vector3(0, 1, 0),
                20.0,
                float(nx)/float(ny),
                aperture,
                dist_to_focus
                );
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
    vector3 cur_attenuation = vector3(1.0, 1.0, 1.0);

    for (int i = 0; i < 50; i++) {
        hit_record rec;

        // while ray keeps bouncing off from objects, iterate
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vector3 attenuation;

            // if the material scatters the ray, track it
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                // otherwise, it's a black body (an object which absorbs all lights)
                return vector3(0.0, 0.0, 0.0);
            }
        } else {
            // ray hit nothing. it's from the background, not an object in the scene
            vector3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vector3 color = (1.0f - t) * vector3(1.0, 1.0, 1.0) + t * vector3(0.5, 0.7, 1.0);
            return cur_attenuation * color;
        }
    }
    return vector3(0.0, 0.0, 0.0);  // exceeded recursion
}

//! \brief Initialize rendering system
//!
//! This function initializes random states allocated for each pixel on frame buffer
//! \param max_x width of the image
//! \param max_y height of the image
//! \param rand_state an array of curandState variables
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x >= max_x) || (y >= max_y)) return;

    unsigned int pixel_index = y * max_x + x;

    // each thread gets same seed, a different sequence number
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vector3* fb, int max_x, int max_y, int num_samples,
                       camera** cam, hittable **world, curandState* rand_state) {
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
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        pixel_color += ray_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    pixel_color /= float(num_samples);
    // gamma correction
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
    checkCudaErrors(cudaMalloc((void**) &d_rand_state, num_pixels * sizeof(curandState)));

    // set world and camera
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void**) &d_list, 5 * sizeof(hittable*)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hittable*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // set dimensions of grid
    int tx = 8;
    int ty = 8;
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << num_samples << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    clock_t start, stop;
    start = clock();
    // render
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, image_width, image_height, num_samples, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::cerr << "took " << timer_seconds << " seconds.\n";

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