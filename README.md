FirstRayTracer
=============
C++ Implementation of basic ray tracer based on the book [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html, "Ray Tracing in One Weekend") by [Peter Shirley](https://github.com/petershirley/home, "Github profile of Peter Shirley") 

![CMake](https://github.com/DveloperY0115/FirstRayTracer/workflows/CMake/badge.svg)

Features
-------------

This simple renderer contains fundamental features of modern physically based renderers such as
- Abstract *vector3* class for holding  & manipulating data of points, colors (RGB), and (obviously) vectors
- Freely movable & rotatable *camera* class whose vertical FOV, and DOF can be adjusted by user
- Frequently used materials: *Lambertian (Diffuse)*, *Metal*, and *Dielectric*
- Utility functions that come in handy for various tasks (random number generation, converting units)
- **A demo code which generates the beautiful image you saw on top of this document**

Plans for Future Updates
-------------
- **CPU multi-threading support**
- **GPU support (will be implemented using CUDA)**
- More geometry
- More materials
- Texture support
- **Command-line support**
- File I/O (to set up large, complex scene within a file and then manage it easily)

Project Structure 
-------------
```
.
├── src                     # Source files
├── .github/workflows       # CI scripts                  
├── .gitignore              # .gitignore file
├── CMakeLists.txt          # CMakeLists.txt file containing project settings & build options                   
├── LICENSE                 # License info of this project
└── README.md               # the file you are reading
```

and under `src` directory there are:
```
.
├── camera.hpp              # *camera* class of FirstRayTracer
├── color.hpp               # a function for writing RGB data to a file
├── hittable.hpp            # *hittable* class of FirstRayTracer
├── hittable_list.hpp       # class for managing multiple *hittable* objects in a scene
├── main.cpp                # a demo code that generates the cover image
├── material.hpp            # general form of *material* class and its derivations
├── ray.hpp                 # *ray* class of FirstRayTracer
├── rtweekend.hpp           # a project header defining constants & utility functions
├── sphere.hpp              # *sphere* class which is a derived class of *hittable*
└── vector3.hpp             # *vector3* class which defines the fundamental ADT of this renderer
```
