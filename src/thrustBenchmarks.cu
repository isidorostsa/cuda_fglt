#include <cuda.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/transform.h>

#include <cstddef>

template<typename T>
std::ostream& operator<<(std::ostream& os, const thrust::host_vector<T>& vec)
{
    os << "|";
    for (const T& el : vec) {
        os << " " << el << " |";
    } os << std::endl;
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<T>& vec)
{
    os << "|";
    for (const T& el : vec) {
        os << " " << el << " |";
    } os << std::endl;
    return os;
}

template<typename T>
__global__ void elwiseMul(T* a, T* b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

#define BLOCK_SIZE 256
template<typename T>
__global__ void adjDif(T* in, T* out, int n){
    __shared__ T temp[BLOCK_SIZE + 1];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x;

    if(gindex < n){
        // Load input into shared memory
        temp[lindex] = in[gindex];
        if(lindex == BLOCK_SIZE-1 || gindex == n-1){
            temp[lindex+1] = in[gindex + 1];
        }
    }

    __syncthreads();

    // Compute the difference
    if(gindex < n)
        out[gindex] = temp[lindex + 1] - temp[lindex];
}

#include <thrust/execution_policy.h>
#include <thrust/random/normal_distribution.h>

#define now std::chrono::high_resolution_clock::now
#define time_point std::chrono::high_resolution_clock::time_point
#define ms_time(a, b) std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count()

int main(int argc, char *argv[])
{

    const int test_size = atoi(argv[1]);
    const size_t blockSize = BLOCK_SIZE;
    const size_t numBlocks = (test_size + blockSize - 1) / blockSize;

    time_point t1, t2;
    {
        thrust::device_vector<float> temp(test_size, 3.0f);
        thrust::device_vector<float> tempres(temp.size());

        // CUDA version:

        std::cout << "Working on vectors of size " << test_size << " with " << numBlocks << " blocks of " << blockSize << " threads each" << std::endl;
        std::cout << "Hadamard Multiplication times:" << std::endl;

        // kernel version
        t1 = now();
        elwiseMul<<<numBlocks, blockSize>>>(temp.data().get(), temp.data().get(), tempres.data().get(), test_size);
        cudaDeviceSynchronize();
        t2 = now();
        std::cout << "\tKernel: " << ms_time(t1, t2) << " ms" << std::endl;

        for(const auto& el: thrust::host_vector<float>(tempres)){
            if(el != 9.0f) {
                std::cout << "\tMistake with Kernel: " << el << std::endl;
                break;
            }
        }


        // thrust version:
        t1 = now();
        thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), tempres.begin(), thrust::multiplies<float>());
        cudaDeviceSynchronize();
        t2 = now();
        std::cout << "\tThrust: " << ms_time(t1, t2) << " ms" << std::endl;

        for(const auto& el: thrust::host_vector<float>(tempres)){
            if(el != 9.0f) {
                std::cout << "\tMistake with Thrust: " << el << std::endl;
                break;
            }
        }
    }
    {
        std::cout << "Differnece:\n";

        thrust::device_vector<float> tempdif(test_size+1, 1.0f);
        thrust::device_vector<float> tempdifres(test_size);

        t1 = now();
        adjDif<<<numBlocks, blockSize>>>(tempdif.data().get(), tempdifres.data().get(), test_size);
        cudaDeviceSynchronize();
        t2 = now();
        std::cout << "\tKernel time taken: " << ms_time(t1, t2) << " ms" << std::endl;
        //std::cout << "Kernel result: " << tempdifres << std::endl;

        t1 = now();
        thrust::adjacent_difference(thrust::device, tempdif.begin(), tempdif.end(), tempdifres.begin());
        cudaDeviceSynchronize();
        t2 = now();
        std::cout << "\tThrust time taken: " << ms_time(t1, t2) << " ms" << std::endl;
        //std::cout << "Thrust result: " << tempdifres << std::endl;
    }
}