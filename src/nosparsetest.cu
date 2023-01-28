#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

__host__ void addVec(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void addVecGPU(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
    //printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d, i: %d\n", blockIdx.x, blockDim.x, threadIdx.x, i);
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    int n = 1 << atoi(argv[1]);
    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];

    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    addVecGPU <<<numBlocks, threadsPerBlock>>> (d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU time: " << elapsed.count() << "s" << std::endl;

/*
    start = std::chrono::high_resolution_clock::now();
    addVec(a, b, c, n);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU time: " << elapsed.count() << "s" << std::endl;
*/
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        maxError = fmax(maxError, fabs(c[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}