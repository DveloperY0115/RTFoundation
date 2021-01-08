//
// Created by dveloperY0115 on 1/8/2021.
//

#include <iostream>

__global__ void hello_from_gpu() {
    printf("Hello world! from gpu [%d, %d]\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello_from_gpu<<<1, 2>>>();

    return 0;
}