#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
#include <chrono>

#include "common/host_structs.hpp"
#include "common/fileio.hpp"

#define PRINT_HPC 1

#define TIME_OP(NAME, OP) \
      T_START = std::chrono::high_resolution_clock::now(); \
      OP; \
      T_END = std::chrono::high_resolution_clock::now(); \
      if(PRINT_HPC) {\
        if (std::string(NAME) == "HOST_TO_DEVICE" || std::string(NAME) == "FGLT" || std::string(NAME) == "DEVICE_TO_HOST")\
            std::cout << (double)std::chrono::duration_cast<std::chrono::microseconds>(T_END-T_START).count() << '\t'; \
      }\
      else std::cout << NAME << " took " << (double)std::chrono::duration_cast<std::chrono::microseconds>(T_END-T_START).count()/1000.0 << " ms" << std::endl;

__global__
void kernel_spmv(int n_vertices, int* A_offsets, int* A_positions, int *x, int *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < n_vertices){
    y[idx] = 0;

    for (int k = A_offsets[idx]; k < A_offsets[idx + 1]; k++) {
      y[idx] += x[A_positions[k]];
    } 
  }
}


__global__
void kernel_c3(int n_vertices, int* A_offsets, int* A_positions, int *c3) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vertices) {
    const int i_nb_start = A_offsets[i];
    const int i_nb_end = A_offsets[i + 1];

    for (int i_nb_idx = i_nb_start; i_nb_idx < i_nb_end; i_nb_idx++) {
      int j = A_positions[i_nb_idx];

      if(i<=j) break;

      const int j_nb_start = A_offsets[j];
      const int j_nb_end = A_offsets[j + 1];

      int _i_nb_idx = i_nb_start;
      int _j_nb_idx = j_nb_start;

      while (_i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end)
      {
        if ((A_positions[_i_nb_idx] > i) || (A_positions[_j_nb_idx] > j)){
            break;
        }
        else if (A_positions[_i_nb_idx] == A_positions[_j_nb_idx])
        {
          c3[j]++;
          c3[i]++;
          c3[A_positions[_i_nb_idx]]++;
          _i_nb_idx++;
          _j_nb_idx++;
        }
        else if (A_positions[_i_nb_idx] < A_positions[_j_nb_idx])
        {
          _i_nb_idx++;
        }
        else
        {
          _j_nb_idx++;
        }
      }
    }
  }
}


__global__
void kernel_s0(int n_vertices, int* s0){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    (s0)[idx] = 1;
  }
}
__global__
void kernel_s1(int n_vertices, int* A_offsets, int* s1){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    s1[idx] = A_offsets[idx + 1] - A_offsets[idx];
  }
}

__global__
void kernel_s2(int n_vertices, int* s1, int* s2){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    s2[idx] -= s1[idx];
  }
}

__global__
void kernel_s3(int n_vertices, int* s1, int* s3){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
    s3[idx] = (s1[idx] * (s1[idx] - 1)) / 2;
  }
}

__global__
void kernel_s4(int n_vertices, int* s2, int* s3, int* s4){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vertices) {
     s2[idx] -= 2 * s4[idx];
     s3[idx] -= s4[idx];
  }
}

void fglt(h_csr* h_A){
  // struct timespec T_START, T_END;
  std::chrono::high_resolution_clock::time_point T_START, T_END;

  std::chrono::high_resolution_clock::time_point FGLT_START, FGLT_END;

  // Allocate device vectors
  TIME_OP("HOST_TO_DEVICE",
    // Allocate host vectors
    int* d_A_offsets;
    int *d_A_positions;
    // Send A to Device
    cudaMalloc(&d_A_offsets, (h_A->get_rows() + 1)* sizeof(int));
    cudaMalloc(&d_A_positions, h_A->get_nnz() * sizeof(int));
    cudaMemcpy(d_A_offsets, h_A->get_offsets().data(), (h_A->get_rows() + 1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_positions, h_A->get_positions().data(), h_A->get_nnz() * sizeof(int), cudaMemcpyHostToDevice);
  )


  // Run calculations on device

  FGLT_START = std::chrono::high_resolution_clock::now(); \

  int *d_d0, *d_d1, *d_d2, *d_d3, *d_d4;
  cudaMalloc(&d_d0, h_A->get_rows() * sizeof(int));
  cudaMalloc(&d_d1, h_A->get_rows() * sizeof(int));
  cudaMalloc(&d_d2, h_A->get_rows() * sizeof(int));
  cudaMalloc(&d_d3, h_A->get_rows() * sizeof(int));
  cudaMalloc(&d_d4, h_A->get_rows() * sizeof(int));



  int blockSize = 1024;
  int numBlocks = (h_A->get_rows() + blockSize - 1) / blockSize;

  TIME_OP("d0",
  (kernel_s0   <<<numBlocks, blockSize>>>(h_A->get_rows(), d_d0));
  cudaDeviceSynchronize();
  );

  TIME_OP("d1",
  (kernel_s1   <<<numBlocks, blockSize>>>(h_A->get_rows(), d_A_offsets , d_d1));
  cudaDeviceSynchronize();
  );

  TIME_OP("d2",
  (kernel_spmv <<<numBlocks, blockSize>>>(h_A->get_rows(), d_A_offsets, d_A_positions, d_d1, d_d2));
  (kernel_s2   <<<numBlocks, blockSize>>>(h_A->get_rows(), d_d1, d_d2));
  cudaDeviceSynchronize();
  );

  TIME_OP("d3",
  (kernel_s3   <<<numBlocks, blockSize>>>(h_A->get_rows(), d_d1, d_d3));
  cudaDeviceSynchronize();
  );
  
  TIME_OP("c3",
  (kernel_c3 <<<numBlocks, blockSize>>>(h_A->get_rows(), d_A_offsets, d_A_positions, d_d4)); 
  cudaDeviceSynchronize();
  );
  
  TIME_OP("d4",
  (kernel_s4   <<<numBlocks, blockSize>>>(h_A->get_rows(), d_d2, d_d3, d_d4));
  cudaDeviceSynchronize();
  );

  FGLT_END = std::chrono::high_resolution_clock::now(); \
    
  if(PRINT_HPC) std::cout << (double)std::chrono::duration_cast<std::chrono::microseconds>(FGLT_END-FGLT_START).count() << '\t'; \

  // Transfer results from device to host
  TIME_OP("DEVICE_TO_HOST",
  int *h_d0;
  int *h_d1;
  int *h_d2;
  int *h_d3;
  int *h_d4;
  h_d0 = (int*)calloc(h_A->get_rows(), sizeof(int));
  h_d1 = (int*)calloc(h_A->get_rows(), sizeof(int));
  h_d2 = (int*)calloc(h_A->get_rows(), sizeof(int));
  h_d3 = (int*)calloc(h_A->get_rows(), sizeof(int));
  h_d4 = (int*)calloc(h_A->get_rows(), sizeof(int));
  cudaMemcpy(h_d0, d_d0, (h_A->get_rows())*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d1, d_d1, (h_A->get_rows())*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d2, d_d2, (h_A->get_rows())*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d3, d_d3, (h_A->get_rows())*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d4, d_d4, (h_A->get_rows())*sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  );
  
  // Free device memory
  cudaFree(d_d0);
  cudaFree(d_d1);
  cudaFree(d_d2);
  cudaFree(d_d3);
  cudaFree(d_d4);
  cudaFree(d_A_offsets);
  cudaFree(d_A_positions);
}



int main(int argc, char *argv[]) {

  // Initialize cuda context
  cudaFree(0);

  // struct timespec T_START, T_END;
  std::chrono::high_resolution_clock::time_point T_START, T_END;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
    exit(1);
  }

  // Read mtx file
  TIME_OP("Loading the file",
    h_csr h_A = loadSymmFileToCsr(argv[1]);
  )

  TIME_OP("The whole fglt",   
    fglt(&h_A);
  );

  std::cout << std::endl;

  return 0;
}