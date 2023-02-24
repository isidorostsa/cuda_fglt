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

#define time_bench(func, name) \
    {\
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();\
    func;\
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();\
    std::cout << "Time taken for " << #name << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;\
    }

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

void get_c3_v2_3(const h_csr &A, int *c3)
{  
  std::fill(c3, c3 + A.get_rows(), 0);

  for (int i = 0; i < A.get_rows(); i++)
  {
    const int i_nb_start = A.offsets[i];
    const int i_nb_end = A.offsets[i + 1];

    for (int i_nb_idx = i_nb_start; i_nb_idx < i_nb_end; i_nb_idx++)
    {
      const int j = A.positions[i_nb_idx];
      if (i < j) break;

      const int j_nb_start = A.offsets[j];
      const int j_nb_end = A.offsets[j + 1];

      int _i_nb_idx = i_nb_start;
      int _j_nb_idx = j_nb_start;

      while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end)
      {
        const int _i_nb_pos = A.positions[_i_nb_idx];
        const int _j_nb_pos = A.positions[_j_nb_idx];

        if(_i_nb_pos > i || _j_nb_pos > j) break;

        if (_i_nb_pos > _j_nb_pos)
        {
          _j_nb_idx++;
        }
        else if (_i_nb_pos < _j_nb_pos)
        {
          _i_nb_idx++;
        }
        else
        {
          c3[i]++, c3[j]++, c3[_i_nb_pos]++,  _i_nb_idx++, _j_nb_idx++;
        }
      }
    }
  }
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



int main(int argc, char *argv[])
{

    std::string filename(argv[1]); 
    // HOST DATA
    h_csr h_A = loadSymmFileToCsr(filename);

    thrust::host_vector<float> h_A_vals(h_A.nnz, 1.0f);

    const int n = h_A.rows;
    const int nnz = h_A.nnz;

    std::cout << "A = " << std::endl;
    printCSR(h_A.offsets, h_A.positions, h_A_vals, n, n, nnz);
    std::cout << "A Sparsity: " << (100 * (1 - ((float)h_A.nnz / ((float)(n) * (float)n)))) << "%" << std::endl;
    std::cout << "A positions: " << h_A.positions.size() << std::endl;

    // DEVICE DATA
    d_cusparse_csr d_A(h_A);

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    /*

                A_2 = A*A COMPUTATION

    d_cusparse_csr d_A2 = d_cusparse_csr::multiply(d_A, d_A, handle);

    std::cout << "A2 = " << std::endl;
    printCSR(d_A2.get_offsets(), d_A2.get_positions(), d_A2.get_values(), n, n, d_A2.get_nnz());
    */

    /*

                CALCULATE p1

    */
    thrust::device_vector<float> d_p1(n);
    // make a copy of d_A_offs but with floats
    thrust::device_vector<float> d_A_offs_float(d_A.get_offsets());

    thrust::transform(
        d_A_offs_float.begin() + 1, d_A_offs_float.end(),
        d_A_offs_float.begin(), // d_A_offs_float.end()-1,
        d_p1.begin(),
        thrust::minus<float>());

    std::cout << "p1: " << d_p1;

    /*

                CALCULATE c3

    */

    thrust::host_vector<int> h_c3(n);
        get_c3_v2_3(h_A, h_c3.data());
    std::cout << "c3: " << h_c3;

        thrust::device_vector<float> d_c3 = h_c3;
    /*

                CALCULATE p2

    */
    thrust::device_vector<float> d_Ap1 = d_cusparse_csr::multiply(d_A, d_p1, handle);


    std::cout << "Ap1: " << d_Ap1;

    thrust::device_vector<float> d_p2(n);

        thrust::transform(
            d_Ap1.begin(), d_Ap1.end(),
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
            d2_trans()        );
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
        d3_trans());
    std::cout << "d3: " << d_d3;

    // DONE

    std::cout << "d1 = " << d_p1;
    std::cout << "d2 = " << d_d2;
    std::cout << "d3 = " << d_d3;
    std::cout << "d4 = " << d_c3;
    return 0;
}