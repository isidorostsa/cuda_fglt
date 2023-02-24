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

__global__ void c3_kernel(const int *offsets, const int *positions, int *c3, int n)
{    
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    const int i_nb_start = offsets[i];
    const int i_nb_end = offsets[i + 1];

    int amt_i = 0;
    for (int i_nb_idx = i_nb_start; i_nb_idx < i_nb_end; i_nb_idx++)
    {
      const int j = positions[i_nb_idx];
      if (i < j) break;

      const int j_nb_start = offsets[j];
      const int j_nb_end = offsets[j + 1];

      int _i_nb_idx = i_nb_start;
      int _j_nb_idx = j_nb_start;

      int amt_j = 0;
      while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end)
      {
        const int _i_nb_pos = positions[_i_nb_idx];
        const int _j_nb_pos = positions[_j_nb_idx];

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
          amt_j++, atomicAdd(&c3[_i_nb_pos], 1),  _i_nb_idx++, _j_nb_idx++;
        }
      }

      amt_i += amt_j;

      atomicAdd(&c3[j], amt_j);
    }

    atomicAdd(&c3[i], amt_i);
  }
}

thrust::device_vector<int> get_c3_v3(const d_cusparse_csr& A) {
    const int n = A.get_rows();

    thrust::device_vector<int> d_c3(n);

    const int *d_A_offs = A.get_offsets().data().get();
    const int *d_A_pos = A.get_positions().data().get();

    int *d_c3_ptr = d_c3.data().get();

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block / 10;

    c3_kernel<<<blocks, threads_per_block>>>(d_A_offs, d_A_pos, d_c3_ptr, n);

    return d_c3;
}

thrust::device_vector<
    thrust::device_vector<float>
> fglt(const h_csr& h_A) {
    const int n = h_A.rows;

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    d_cusparse_csr d_A(h_A);
    /*
                CALCULATE p1
    */
    thrust::device_vector<float> d_p1(n);
    // make a copy of d_A_offs but with floats, because cusparse only accepts floats
    thrust::device_vector<float> d_A_offs_float(d_A.get_offsets());

    // adjecent difference
    thrust::transform(
        d_A_offs_float.begin() + 1, d_A_offs_float.end(),
        d_A_offs_float.begin(), // d_A_offs_float.end()-1,
        d_p1.begin(),
        thrust::minus<float>()
    );

    /*
                CALCULATE c3
    */

    thrust::device_vector<int> d_c3_int = get_c3_v3(d_A);
    thrust::device_vector<float> d_c3(d_c3_int);

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
        d2_trans()
    );

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

    return {};
}

thrust::device_vector<
    thrust::device_vector<float>
> fglt(const d_cusparse_csr& d_A) {
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    const int n = d_A.get_rows();
    /*
                CALCULATE p1
    */
    thrust::device_vector<float> d_p1(n);
    // make a copy of d_A_offs but with floats, because cusparse only accepts floats
    thrust::device_vector<float> d_A_offs_float(d_A.get_offsets());

    // adjecent difference
    thrust::transform(
        d_A_offs_float.begin() + 1, d_A_offs_float.end(),
        d_A_offs_float.begin(), // d_A_offs_float.end()-1,
        d_p1.begin(),
        thrust::minus<float>()
    );

    /*
                CALCULATE c3
    */

    thrust::device_vector<int> d_c3_int = get_c3_v3(d_A);
    thrust::device_vector<float> d_c3(d_c3_int);

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
        d2_trans()
    );

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

    const d_cusparse_csr d_A(h_A);
    t1 = std::chrono::high_resolution_clock::now();
    fglt(d_A);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken to compute the fglt: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}