ray-tracing-the-next-week-cpp
=============

![cover_image](mediamg-1.21-book1-final.jpg)
C++ Implementation of basic ray tracer based on the book [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html, "Ray Tracing: The Next Week") by [Peter Shirley](https://github.com/petershirley/home, "Github profile of Peter Shirley"). This project is written in C++ 14 and can be compiled with popular compiler systems such as Clang, MSVC, etc.

[![Windows](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/windows.yml/badge.svg)](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/windows.yml)
[![Windows-CUDA](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/windows-cuda.yml/badge.svg)](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/windows-cuda.yml)
[![Ubuntu](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/ubuntu.yml)
[![Ubuntu-CUDA](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/ubuntu-cuda.yml/badge.svg)](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/ubuntu-cuda.yml)
[![macOS](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/macos.yml/badge.svg)](https://github.com/DveloperY0115/ray-tracing-in-one-weekend-cpp/actions/workflows/macos.yml)

Features
-------------

This simple renderer contains fundamental features of modern physically based renderers such as
- Abstract *Vector3* class for holding  & manipulating data of points, colors (RGB), and (obviously) vectors
- Freely movable & rotatable *Camera* class whose vertical FOV, and DOF can be adjusted by user
- Frequently used materials: *Lambertian (Diffuse)*, *Metal*, and *Dielectric*
- Utility functions that come in handy for various tasks (random number generation, converting units)
- **A demo code which generates the beautiful image you saw on top of this document**

Also, it **supports GPU accelerated rendering on Windows having CUDA 10 or higher.**

Plans for Future Updates
-------------
- [x] **CPU multi-threading support (OpenMP)**
- [x] **GPU support (CUDA)**
- [ ] More geometry
- [ ] More materials
- [ ] Texture support
- [ ] **Command-line support**
- [ ] File I/O (to set up large, complex scene within a file and then manage it easily)

Project Structure 
-------------
```
.
├── src                     # Source files
    ├── cpu                 # CPU version
    ├── gpu                 # GPU version (Requires CUDA 10.1 or higher)
├── .github/workflows       # CI scripts                 
├── .gitignore              # .gitignore file
├── CMakeLists.txt          # CMakeLists.txt file containing project settings & build options                   
├── LICENSE                 # License info of this project
└── README.md               # the file you are reading
```

under `src/cpu` directory there are:
```
.
├── Camera.hpp              # Camera class of FirstRayTracer
├── Color.hpp               # a function for writing RGB data to a file
├── dielectric.hpp          # Dielectric material class
├── hittable.hpp            # hittable class of FirstRayTracer
├── hittable_list.hpp       # class for managing multiple hittable objects in a scene
├── lambertian.hpp          # Lambertian (diffuse) material class
├── main.cpp                # a demo code that generates the cover image
├── material.hpp            # general form of material class
├── metal.hpp               # Metal material class
├── ray.hpp                 # ray class of FirstRayTracer
├── rtweekend.hpp           # a project header defining constants & utility functions
├── sphere.hpp              # sphere class which is a derived class of hittable
└── Vector3.hpp             # Vector3 class which defines the fundamental ADT of this renderer
```

and most of files under `src/gpu` are written based on the CPU implementation, thus structure and design principle are almost identical.
