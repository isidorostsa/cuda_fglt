#include <cuda.h>
#include <cusparse.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<typename T>
std::ostream& operator<<(std::ostream& os, const thrust::host_vector<T>& vec)
{
    os << "|";
    for (const T& el : vec) {
        os << " " << el << " |";
    } os << std::endl;
    return os;
}

#define threads_per_block 1024

template <typename T>
__global__ void mulTest(T* a, T* b, T* c){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] * b[idx];
}

template <typename T>
double testType(int blocks){
    const int size = blocks * threads_per_block;

    T *a, *b, *c;

    cudaMalloc(&a, size * sizeof(T));
    cudaMalloc(&b, size * sizeof(T));
    cudaMalloc(&c, size * sizeof(T));

    // fill a and b with random values
    T* h_a = new T[size];
    T* h_b = new T[size];

    for(int i = 0; i < size; i++){
        h_a[i] = float(rand() % 100)/100;
        h_b[i] = float(rand() % 100)/100;
    }

    cudaMemcpy(a, h_a, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, size * sizeof(T), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 50; i++){
        mulTest<<<blocks, threads_per_block>>>(a, b, c);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    // result in milliseconds
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/50;
}

int main(int argc, char *argv[])
{
    int blocks = atoi(argv[1]);
    float size = blocks * threads_per_block;

    std::cout << "Size: " << size << std::endl;

    std::cout << "char: " << (size/testType<char>(blocks)) << "*10^9 Char Operations per second" << std::endl;
    std::cout << "short: " << (size/testType<short>(blocks)) << "*10^9 Short Operations per second" << std::endl;
    std::cout << "int: " << (size/testType<int>(blocks)) << "*10^9 Int Operations per second" << std::endl;
    std::cout << "long: " << (size/testType<long>(blocks)) << "*10^9 Long Operations per second" << std::endl;
    std::cout << "float: " << (size/testType<float>(blocks)) << "*10^9 Float Operations per second" << std::endl;
    std::cout << "double: " << (size/testType<double>(blocks)) << "*10^9 Double Operations per second" << std::endl;
}
