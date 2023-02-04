#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void const_write(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        (*c) += a[i];
    }
}

__global__ void rand_both(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] += a[i];
    }
}

__global__ void const_read(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += *c;
    }
}

int main() {
    // is it faster to write to a constant location while reading from random locations
    // or to write to random locations while reading from a constant location?

    int n = 1 << 28;

    thrust::host_vector<int> h_a(n, 23);
    thrust::host_vector<int> h_b(n, 14);

    thrust::device_vector<int> d_a = h_a;
    thrust::device_vector<int> d_b = h_b;

    int *d_c;
    cudaMalloc(&d_c, sizeof(int));

    int *h_c = (int *)malloc(sizeof(int));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    const_write<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), d_c, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "const write: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    rand_both<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), d_c, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "rand both: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    const_read<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), d_c, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "const read: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "us" << std::endl;

    cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "c: " << *h_c << std::endl;

    cudaFree(d_c);
    free(h_c);

    return 0;
}