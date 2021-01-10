//
// Created by dveloperY0115 on 1/9/2021.
//

#include <iostream>
#include "vector3.hpp"

__global__ void hello_from_gpu() {
    printf("Hello from [%d, %d]\n", threadIdx.x, blockIdx.x);
}

int main() {
    vector3 vec1 = vector3();
}