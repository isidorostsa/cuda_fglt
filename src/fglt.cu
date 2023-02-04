#include <cuda.h>
#include <cusparse.h>
#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/transform.h>

#include <cstddef>

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

#define computeType CUDA_R_32F

struct d3_trans
{
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr float operator()(const float &lhs, const float &rhs) const
  {
    return (lhs)*(lhs-1) - 2*rhs;
  }
};

int main(int argc, char *argv[])
{

    // HOST DATA
    h_csr h_A = loadFileToCsr("/work_dir/datasets/test.mtx");
    thrust::host_vector<float> h_A_vals(h_A.nnz, 1.0f);

    const int n = h_A.rows;
    const int nnz = h_A.nnz;

    std::cout << "A = " << std::endl;
    printCSR(h_A.offsets, h_A.positions, h_A_vals, n, n, nnz);
    std::cout << "A Sparsity: " << (100 * ( 1 - ((float)h_A.nnz / (float)(n * n)))) << "%" << std::endl;
    std::cout << "A positions: " << h_A.positions.size() << std::endl;

    // DEVICE DATA
    d_cusparse_csr d_A(h_A);

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))
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

    // const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // adjDif<<<num_blocks, BLOCK_SIZE>>>(d_A_offs_float.data().get(), d_p1.data().get(), n);

    thrust::transform(
        d_A_offs_float.begin() + 1, d_A_offs_float.end(),
        d_A_offs_float.begin(), //d_A_offs_float.end()-1,
        d_p1.begin(),
        thrust::minus<float>()
    );

    std::cout << "p1: " << d_p1;

// DONE 
/*

        HADAMARD PRODUCT OF A2 * A

*/
                // Copy A2 to host
    h_csr h_A2(d_A2);
                // Create space for hadamard product
    h_csr h_C3 = h_csr::hadamard(h_A, h_A2);

    // printCSR 
    std::cout << "C3 = " << std::endl;
    printCSR(h_C3.offsets, h_C3.positions, h_C3.values, n, n, h_C3.nnz);

    d_cusparse_csr d_C3(h_C3);

// DONE

/*

            CALCULATE c3

*/
    thrust::device_vector<float> d_e(n, 1.0f);
    thrust::device_vector<float> d_c3 = d_cusparse_csr::multiply(d_C3, d_e, handle, 0.5f, 0.0f); 

    std::cout << "c3: " << d_c3;
/*

            CALCULATE p2

*/
    thrust::device_vector<float> d_Ap1 = d_cusparse_csr::multiply(d_A, d_p1, handle);

    thrust::device_vector<float> d_p2(n);

    thrust::transform(
        d_Ap1.begin(),
        d_Ap1.end(),
        d_p1.begin(),
        d_p2.begin(),
        thrust::minus<float>()
    );

    std::cout << "p2: " << d_p2;
/*

            CALCULATE d2

*/

    thrust::device_vector<float> d_d2(n);

    thrust::transform(
        d_p2.begin(),
        d_p2.end(),
        d_c3.begin(),
        d_d2.begin(),
        thrust::minus<float>()
    );

    std::cout << "d2: " << d_d2;


/*

            CALCULATE d3

*/

    thrust::device_vector<float> d_d3(n);

    thrust::transform(
        d_p1.begin(),
        d_p1.end(),
        d_c3.begin(),
        d_d3.begin(),
        d3_trans()
    );

    std::cout << "d3: " << d_d3;

// DONE

    std::cout << "d1 = " << d_p1;
    std::cout << "d2 = " << d_d2;
    std::cout << "d3 = " << d_d3;
    std::cout << "d4 = " << d_c3;
    return 0;
}