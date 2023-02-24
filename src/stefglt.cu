#include <iostream>
#include <chrono>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/fileio.hpp"

#include "common/device_csr_wrapper.hpp"
#include "common/host_structs.hpp"

template <typename F>
int* run_test(const std::string &name, F f, const h_csr &A)
{
  std::cout << name << " : ";
  std::cout.flush();
  int *c3 = new int[A.get_rows()];
  std::fill(c3, c3 + A.get_rows(), 0);
  auto start = std::chrono::high_resolution_clock::now();
  f(A, c3);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;
  return c3;
}

void get_c3_v1(const h_csr &A, int *c3)
{
  int j, k, l, lb, up, clb, cup, llb;
  for (int i = 0; i < A.get_rows(); i++)
  {
    lb = A.offsets[i];
    up = A.offsets[i + 1];
    for (j = lb; j < up; j++)
    {
      clb = A.offsets[A.positions[j]];
      cup = A.offsets[A.positions[j] + 1];
      llb = lb;

      for (k = clb; k < cup; k++)
      {
        for (l = llb; l < up; l++)
        {
          if (A.positions[k] == A.positions[l])
          {
            c3[i]++;
            llb = l + 1;
            break;
          }
          else if (A.positions[k] < A.positions[l])
          {
            llb = l;
            break;
          }
          else
          {
            llb = l + 1;
          }
        }
      }
    }
    c3[i] /= 2;
  }
}

void get_c3_v2_1(const h_csr &A, int *c3)
{
  for (int i = 0; i < A.get_rows(); i++)
  {
    const int i_nb_start = A.offsets[i];
    const int i_nb_end = A.offsets[i + 1];

    for (int i_nb_idx = i_nb_start; i_nb_idx < i_nb_end; i_nb_idx++)
    {
      int common_nb = 0;      

      const int j = A.positions[i_nb_idx];

      const int j_nb_start = A.offsets[j];
      const int j_nb_end = A.offsets[j + 1];

      int _i_nb_idx = i_nb_idx;
      int _j_nb_idx = j_nb_start;

      while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end)
      {
        if (A.positions[_i_nb_idx] == A.positions[_j_nb_idx])
        {
          common_nb++, _i_nb_idx++, _j_nb_idx++;
        }
        else if (A.positions[_i_nb_idx] < A.positions[_j_nb_idx])
        {
          _i_nb_idx++;
        }
        else
        {
          _j_nb_idx++;
        }
      }
      c3[i] += common_nb;
    }
  }
}

void get_c3_v2_binary(const h_csr &A, int *c3)
{
  for (int i = 0; i < A.get_rows(); i++)
  {
    const int i_nb_start = A.offsets[i];
    const int i_nb_end = A.offsets[i + 1];

    for (int i_nb_idx = i_nb_start; i_nb_idx < i_nb_end; i_nb_idx++)
    {
      int common_nb = 0;      

      const int j = A.positions[i_nb_idx];

      const int j_nb_start = A.offsets[j];
      const int j_nb_end = A.offsets[j + 1];

      int _i_nb_idx = i_nb_start;
      int _j_nb_idx = j_nb_start;

      for(int _i_nb_idx = i_nb_start; _i_nb_idx < i_nb_end; _i_nb_idx++)
      {
        const int i_nb = A.positions[_i_nb_idx];
        while (_j_nb_idx < j_nb_end && A.positions[_j_nb_idx] < i_nb)
        {
          _j_nb_idx++;
        }
        if (_j_nb_idx < j_nb_end && A.positions[_j_nb_idx] == i_nb)
        {
          common_nb++;
        }
      }
      c3[i] += common_nb;
    }
  }

  std::transform(c3, c3 + A.get_rows(), c3, [](int x) { return x / 2; });
}

void get_c3_v2_2(const h_csr &A, int *c3)
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

      int common_nb = 0;

      const int j_nb_start = A.offsets[j];
      const int j_nb_end = A.offsets[j + 1];

      int _i_nb_idx = i_nb_start;
      int _j_nb_idx = j_nb_start;

      while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end)
      {
        if (A.positions[_i_nb_idx] == A.positions[_j_nb_idx])
        {
          common_nb++, _i_nb_idx++, _j_nb_idx++;
        }
        else if (A.positions[_i_nb_idx] < A.positions[_j_nb_idx])
        {
          _i_nb_idx++;
        }
        else
        {
          _j_nb_idx++;
        }
      }
      c3[j] += common_nb;
      c3[i] += common_nb;
    }
  }

  for(int i = 0; i < A.get_rows(); i++){
    c3[i] /= 2;
  }
}

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

void get_c3_v2_4(const h_csr &A, int *c3)
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

      const int i_nb_size = i_nb_end - i_nb_start;
      const int j_nb_size = j_nb_end - j_nb_start;

      if (i_nb_size > j_nb_size)
      {
        const int low = _j_nb_idx;
        const int high = j_nb_end;

        while (_i_nb_idx < i_nb_end)
        {
          const int _i_nb_pos = A.positions[_i_nb_idx];
          if(_i_nb_pos > i || _i_nb_pos > j) break;

          const int mid = (low + high) / 2;
          if (A.positions[mid] == _i_nb_pos)
          {
            c3[i]++, c3[j]++, c3[_i_nb_pos]++,  _i_nb_idx++, _j_nb_idx++;
          }
          else if (A.positions[mid] < _i_nb_pos)
          {
            _j_nb_idx = mid + 1;
          }
          else
          {
            _j_nb_idx = mid;
          }
        }

      }
    }
  }
}





__global__ void inner_loop(const int i, const int i_nb_start, const int i_nb_end, const int *positions, const int *offsets, int *c3)
{
  const int thread_num = blockIdx.x * blockDim.x + threadIdx.x;
  const int i_nb_idx = i_nb_start + thread_num;

  if(i == 0) printf("HERE");

  if (i_nb_idx < i_nb_end){
    const int j = positions[i_nb_idx];
    const int j_nb_start = offsets[j];
    const int j_nb_end = offsets[j + 1];

    int _i_nb_idx = i_nb_idx;
    int _j_nb_idx = j_nb_start;
    int amt = 0;

    while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end) {
      if (positions[_i_nb_idx] == positions[_j_nb_idx]) {
        amt++, _i_nb_idx++, _j_nb_idx++;
      } else if (positions[_i_nb_idx] < positions[_j_nb_idx]) {
        _i_nb_idx++;
      } else {
        _j_nb_idx++;
      }
    }

    atomicAdd(&c3[i], amt);
  }

}


#define THREADS_PER_BLOCK 256

__global__ void get_c3_v3(const int *offsets, const int *positions, int *c3, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    const int i_nb_start = offsets[i];
    const int i_nb_end = offsets[i + 1];

    if (i == 0 || i == 1)
    {
      printf("HERE");
    }


    int num_threads = i_nb_end - i_nb_start;
    int num_blocks = (num_threads - 1) / THREADS_PER_BLOCK + 1;

    // launch inner loop and wait for it to finish using a stream
    inner_loop<<<num_blocks, THREADS_PER_BLOCK>>>(i, i_nb_start, i_nb_end, positions, offsets, c3);
  }
}

// we do not want to launch more than 1000 kernels at a time
__global__ void get_c3_v3_2(const int *offsets, const int *positions, int *c3, int n)
{
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
    const int i_nb_start = offsets[i];
    const int i_nb_end = offsets[i + 1];

    int num_threads = i_nb_end - i_nb_start;
    int num_blocks = (num_threads - 1) / THREADS_PER_BLOCK + 1;

    // launch inner loop and wait for it to finish using a stream

    // we do not want to launch more than 1000 kernels at a time
      inner_loop<<<num_blocks, THREADS_PER_BLOCK>>>(i, i_nb_start, i_nb_end, positions, offsets, c3);
      // cudaDeviceSynchronize();
      __syncthreads();
  }
}

int main(int argc, char **argv)
{
  std::string filename(argv[1]);
  // HOST DATA
  h_csr h_A = loadSymmFileToCsr(filename);

  int avg_nb = (float)h_A.get_nnz() / h_A.get_rows();

  std::cout << "Running on " << filename << std::endl;

  std::cout << "Megabytes of rows: " << h_A.get_rows() * sizeof(int) / 1024.0 / 1024.0 << "mb" << std::endl;
  std::cout << "Megabytes of nonzeros: " << h_A.get_nnz() * sizeof(int) / 1024.0 / 1024.0 << "mb" << std::endl;
  std::cout << "Average neighbors: " << avg_nb << std::endl;

  // DEVICE DATA
  d_cusparse_csr d_A(h_A);
  thrust::device_vector<int> d_c3(h_A.get_rows(), 0);
  const int num_blocks = (h_A.get_rows() + 1023) / 1024;

  // CPU
  int* c3_v1 = run_test("V1", get_c3_v1, h_A);
  int* c3_v2_2 = run_test("V2_2", get_c3_v2_2, h_A);
  // int *c3_v2_4 = run_test("V2_4", get_c3_v2_4, h_A);
  int *c3_binary = run_test("Binary", get_c3_v2_binary, h_A);

  // GPU

  // std::cout << "Launching kernel with " << num_blocks << " blocks" << std::endl;
  // t1 = std::chrono::high_resolution_clock::now();
  // get_c3_v3_2<<<num_blocks, 1024>>>(d_A.get_offsets().data().get(), d_A.get_positions().data().get(), d_c3.data().get(), h_A.get_rows());
  // cudaDeviceSynchronize();
  // t2 = std::chrono::high_resolution_clock::now();
  // time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  // std::cout << "GPU time: " << time_span.count() << " seconds." << std::endl;


  bool error = false;

  for(int i = 0; i < h_A.get_rows(); i++)
  {
    if (c3_v1[i] != c3_v2_2[i] || c3_v1[i] != c3_v2_4[i] || c3_v1[i] != c3_binary[i])
    {
      std::cout << "Error at " << i << std::endl;
      std::cout << "c3_v1: " << c3_v1[i] << std::endl;
      std::cout << "c3_v2_2: " << c3_v2_2[i] << std::endl;
      std::cout << "c3_v2_4: " << c3_v2_4[i] << std::endl;
      std::cout << "c3_binary: " << c3_binary[i] << std::endl;
      error = true;
      break;
    }
  }

  if (!error)
  {
    std::cout << "No errors" << std::endl;
  }

  return 0;
}