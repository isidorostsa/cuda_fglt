#include <iostream>
#include <chrono>

#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "common/fileio.hpp"
#include "common/printing.hpp"
#include "common/device_csr.hpp"

#define time_op(name, op) \
        start = std::chrono::high_resolution_clock::now(); \
        op; \
        end = std::chrono::high_resolution_clock::now(); \
        std::cout << name << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;

struct d3_trans
{
    __thrust_exec_check_disable__
        __host__ __device__ constexpr COMPUTE_TYPE
        operator()(const COMPUTE_TYPE &lhs, const COMPUTE_TYPE &rhs) const
    {
        return (lhs) * (lhs - 1) / 2 -  rhs;
    }
};

struct d2_trans
{
    __thrust_exec_check_disable__
        __host__ __device__ constexpr COMPUTE_TYPE
        operator()(const COMPUTE_TYPE &lhs, const COMPUTE_TYPE &rhs) const
    {
        return (lhs) - 2 * rhs;
    }
};

__global__ void c3_kernel(const int *offsets, const int *positions, COMPUTE_TYPE *c3, int n)
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

thrust::device_vector<COMPUTE_TYPE> get_c3_v3(const d_csr& A) {
    const int n = A.get_rows();

    thrust::device_vector<COMPUTE_TYPE> d_c3(n);

    const int *d_A_offs = A.get_offsets().data().get();
    const int *d_A_pos = A.get_positions().data().get();

    COMPUTE_TYPE *d_c3_ptr = d_c3.data().get();

    const int SmSize = 16;
    const int threadsPerBlock = 256;
    const int threadsPerSM = threadsPerBlock * SmSize;

    const int FullSMs = (n + threadsPerSM - 1) / threadsPerSM;

    c3_kernel<<<SmSize*FullSMs, threadsPerBlock>>>(d_A_offs, d_A_pos, d_c3_ptr, n);
    // c3_kernel<<<1, 1>>>(d_A_offs, d_A_pos, d_c3_ptr, n);
    cudaDeviceSynchronize();

    return d_c3;
}

thrust::host_vector<
    thrust::device_vector<COMPUTE_TYPE>
> fglt(const d_csr& d_A) {

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;

    const int n = d_A.get_rows();
    /*
                CALCULATE p1
    */
time_op("p1",
    thrust::device_vector<COMPUTE_TYPE> d_p1(n);
    // adjecent difference
    thrust::transform(
        d_A.offsets.begin() + 1, d_A.offsets.end(),
        d_A.offsets.begin(), // d_A_offs_COMPUTE_TYPE.end()-1,
        d_p1.begin(),
        thrust::minus<COMPUTE_TYPE>()
    )
)
    /*
                CALCULATE c3
    */

time_op("c3",
    thrust::device_vector<COMPUTE_TYPE> d_c3 = get_c3_v3(d_A);
)

    /*
                CALCULATE p2
    */
time_op("p2",
    thrust::device_vector<COMPUTE_TYPE> d_Ap1 = d_csr::spmv_symbolic(d_A, d_p1);
    thrust::device_vector<COMPUTE_TYPE> d_p2(n);
    thrust::transform(
        d_Ap1.begin(), d_Ap1.end(),
        d_p1.begin(),
        d_p2.begin(),
        thrust::minus<COMPUTE_TYPE>()
    );
)
    /*
                CALCULATE d2
    */
time_op("d2",
    thrust::device_vector<COMPUTE_TYPE> d_d2(n);
    thrust::transform(
        d_p2.begin(),
        d_p2.end(),
        d_c3.begin(),
        d_d2.begin(),
        d2_trans()
    );
)
    /*
                CALCULATE d3
    */
time_op("d3",
    thrust::device_vector<COMPUTE_TYPE> d_d3(n);
    thrust::transform(
        d_p1.begin(),
        d_p1.end(),
        d_c3.begin(),
        d_d3.begin(),
        d3_trans()
    );
)

    cudaDeviceSynchronize();

time_op("return vectors creation",
    thrust::host_vector<thrust::device_vector<COMPUTE_TYPE>> return_vector(4);
    return_vector.push_back(std::move(d_p1));
    return_vector.push_back(std::move(d_p2));
    return_vector.push_back(std::move(d_d3));
    return_vector.push_back(std::move(d_c3));
)
    return return_vector;
}

int main(int argc, char *argv[])
{
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;

    if(argc != 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    std::string filename(argv[1]); 

    time_op("Loading the file",
        const h_csr h_A = loadSymmFileToCsr(filename);
    )

    std::cout << "A = \n";
    printCSR(h_A);

    time_op("Moving the matrix to the device",
        const d_csr d_A(h_A);
    )

    time_op("The whole fglt",
        thrust::host_vector<thrust::device_vector<COMPUTE_TYPE>> h_fglt = fglt(d_A);
    )
}