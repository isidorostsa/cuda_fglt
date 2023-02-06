#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cusparse.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "common/fileio.hpp"
#include "common/sparse_funcs.hpp"
#include "common/printing.hpp"
#include "common/device_csr_wrapper.hpp"

#define CHECK_CUDA(call)                                               \
    {                                                                  \
        cudaError_t status = (call);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            exit(1);                                                   \
        }                                                              \
    }

#define CHECK_CUSPARSE(call)                                                                           \
    {                                                                                                  \
        cusparseStatus_t status = call;                                                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                         \
        {                                                                                              \
            fprintf(stderr, "cuSparse error %s in file '%s' in line %i : %s.\n",                       \
                    cusparseGetErrorName(status), __FILE__, __LINE__, cusparseGetErrorString(status)); \
            exit(1);                                                                                   \
        }                                                                                              \
    }
struct d3_trans
{
    __thrust_exec_check_disable__
        __host__ __device__ constexpr float
        operator()(const float &lhs, const float &rhs) const
    {
        return (lhs) * (lhs - 1) / 2 -  rhs;
    }
};

struct d2_trans
{
    __thrust_exec_check_disable__
        __host__ __device__ constexpr float
        operator()(const float &lhs, const float &rhs) const
    {
        return (lhs) - 2 * rhs;
    }
};
thrust::device_vector<float> fglt(const h_csr& h_A) {
    const int n = h_A.rows;

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    d_cusparse_csr d_A(h_A);
    /*
                A_2 = A*A COMPUTATION
    */
    d_cusparse_csr d_A2 = d_cusparse_csr::multiply(d_A, d_A, handle);
    /*
                CALCULATE p1
    */
    thrust::device_vector<float> d_p1(n);
    // make a copy of d_A_offs but with floats
    thrust::device_vector<float> d_A_offs_float(d_A.get_offsets());

    // adjecent difference
    thrust::transform(
        d_A_offs_float.begin() + 1, d_A_offs_float.end(),
        d_A_offs_float.begin(), // d_A_offs_float.end()-1,
        d_p1.begin(),
        thrust::minus<float>()
    );

    /*
            HADAMARD PRODUCT OF A2 * A
    */
    h_csr h_A2(d_A2);
    h_csr h_C3 = h_csr::hadamard(h_A, h_A2);

    // copy to device
    d_cusparse_csr d_C3(h_C3);
    /*
                CALCULATE c3
    */
    thrust::device_vector<float> d_e(n, 1.0f);
    thrust::device_vector<float> d_c3 = d_cusparse_csr::multiply(d_C3, d_e, handle, 0.5f, 0.0f);
    /*
                CALCULATE p2
    */
    thrust::device_vector<float> d_Ap1 = d_cusparse_csr::multiply(d_A, d_p1, handle);

    thrust::device_vector<float> d_p2(n);
    thrust::transform(
        d_Ap1.begin(), d_Ap1.end(),
        d_p1.begin(),
        d_p2.begin(),
        thrust::minus<float>()
    );
    /*
                CALCULATE d2
    */
    thrust::device_vector<float> d_d2(n);
    thrust::transform(
        d_p2.begin(),
        d_p2.end(),
        d_c3.begin(),
        d_d2.begin(),
        d2_trans());
    /*
                CALCULATE d3
    */
    thrust::device_vector<float> d_d3(n);
    thrust::transform(
        d_p1.begin(),
        d_p1.end(),
        d_c3.begin(),
        d_d3.begin(),
        d3_trans());

    std::cout << "d1 = " << d_p1;
    std::cout << "d2 = " << d_d2;
    std::cout << "d3 = " << d_d3;
    std::cout << "d4 = " << d_c3;

    return {};
}

#define computeType CUDA_R_32F

int main(int argc, char *argv[])
{
    if(argc != 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    std::string filename(argv[1]); 

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const h_csr h_A = loadSymmFileToCsr(filename);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken to load the file: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    std::cout << "A = \n";
    printCSR(h_A);

    t1 = std::chrono::high_resolution_clock::now();
    fglt(h_A);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken to compute the fglt: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}