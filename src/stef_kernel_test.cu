#include <iostream>
#include <chrono>

#include "common/fileio.hpp"

#include "common/device_csr_wrapper.hpp"
#include "common/host_structs.hpp"

void get_c3(h_csr &A, int *c3)
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

__global__ void c3_kernel(const int *offsets, const int *positions, int rows, int *c3)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < rows)
  {
    int lb = offsets[i];
    int up = offsets[i + 1];
    for (int j = lb; j < up; j++)
    {
      int clb = offsets[positions[j]];
      int cup = offsets[positions[j] + 1];
      int llb = lb;
      for (int k = clb; k < cup; k++)
      {
        for (int l = llb; l < up; l++)
        {
          if (positions[k] == positions[l])
          {
            c3[i]++;
            llb = l + 1;
            break;
          }
          else if (positions[k] < positions[l])
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

int main(int argc, char **argv)
{
  std::string filename(argv[1]);
  // HOST DATA
  h_csr h_A = loadSymmFileToCsr(filename);

  // DEVICE DATA
  d_cusparse_csr d_A(h_A);

  std::cout << "Number of rows: " << h_A.get_rows() << std::endl;
  std::cout << "Number of nonzeros: " << h_A.get_nnz() << std::endl;

  // CPU
  int *c3 = new int[h_A.get_rows()];
  memset(c3, 0, h_A.get_rows() * sizeof(int));

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  get_c3(h_A, c3);
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "CPU time: " << time_span.count() << " seconds." << std::endl;

  // GPU
  std::cout << "GPU" << std::endl;
  std::cout << "Allocating " << h_A.get_rows() * sizeof(int) / 1024.0 / 1024.0 << "mb" << std::endl;
  int *d_c3;
  cudaMalloc(&d_c3, h_A.get_rows() * sizeof(int));
  cudaMemset(d_c3, 0, h_A.get_rows() * sizeof(int));
  
  std::cout << "Launching kernel" << std::endl;

  t1 = std::chrono::high_resolution_clock::now();
  c3_kernel<<<(h_A.get_rows() + 1023) / 1024, 1024>>>(d_A.get_offsets().data().get(), d_A.get_positions().data().get(), h_A.get_rows(), d_c3);
  cudaDeviceSynchronize();
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "GPU time: " << time_span.count() << " seconds." << std::endl;

  // Compare
  int *h_c3 = new int[h_A.get_rows()];
  cudaMemcpy(h_c3, d_c3, h_A.get_rows() * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_A.get_rows(); i++)
  {
    if (c3[i] != h_c3[i])
    {
      std::cout << "Error at " << i << std::endl;
      std::cout << "CPU: " << c3[i] << std::endl;
      std::cout << "GPU: " << h_c3[i] << std::endl;
    }
  }

  return 0;
}