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



void spmv(int n_vertices, int* A_offsets, int* A_positions, int *x, int *y) {
  #pragma omp parallel for
  for (int i=0; i<n_vertices; i++){
    y[i] = 0;

    for (int k = A_offsets[i]; k < A_offsets[i + 1]; k++) {
      y[i] += x[A_positions[k]];
    } 
  }
}



void c3(int n_vertices, int* A_offsets, int* A_positions, int *c3) {
  #pragma omp parallel for
  for(int i=0; i<n_vertices; i++){
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
          #pragma omp atomic
          c3[j]++;
          #pragma omp atomic
          c3[i]++;
          #pragma omp atomic
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



void s0(int n_vertices, int* s0){
  #pragma omp parallel for
  for(int i=0; i<n_vertices; i++){
    (s0)[i] = 1;
  }
}

void s1(int n_vertices, int* A_offsets, int* s1){
  #pragma omp parallel for
  for(int i=0; i<n_vertices; i++){
    s1[i] = A_offsets[i + 1] - A_offsets[i];
  }
}


void s2(int n_vertices, int* s1, int* s2){
  #pragma omp parallel for
  for(int i=0; i<n_vertices; i++){
    s2[i] -= s1[i];
  }
}


void s3(int n_vertices, int* s1, int* s3){
  #pragma omp parallel for
  for(int i=0; i<n_vertices; i++){
    s3[i] = (s1[i] * (s1[i] - 1)) / 2;
  }
}


void s4(int n_vertices, int* s2, int* s3, int* s4){
  #pragma omp parallel for
  for(int i=0; i<n_vertices; i++){
    s2[i] -= 2 * s4[i];
    s3[i] -= s4[i];
  }
}

void fglt(h_csr* h_A){
  // struct timespec T_START, T_END;
  std::chrono::high_resolution_clock::time_point T_START, T_END;

  // Allocate host vectors
  int *d0, *d1, *d2, *d3, *d4;
  d0 = (int*)calloc(h_A->get_rows(), sizeof(int));
  d1 = (int*)calloc(h_A->get_rows(), sizeof(int));
  d2 = (int*)calloc(h_A->get_rows(), sizeof(int));
  d3 = (int*)calloc(h_A->get_rows(), sizeof(int));
  d4 = (int*)calloc(h_A->get_rows(), sizeof(int));
 

  int* A_offsets = h_A->offsets.data();
  int* A_positions = h_A->positions.data();



  TIME_OP("d0",
  s0(h_A->get_rows(), d0);
  );

  TIME_OP("d1",
  s1(h_A->get_rows(), A_offsets , d1);
  );

  TIME_OP("d2",
  spmv(h_A->get_rows(), A_offsets, A_positions, d1, d2);
  s2(h_A->get_rows(), d1, d2);
  );

  TIME_OP("d3",
  s3(h_A->get_rows(), d1, d3);
  );
  
  TIME_OP("c3",
  c3(h_A->get_rows(), A_offsets, A_positions, d4); 
  );
  
  TIME_OP("d4",
  s4(h_A->get_rows(), d2, d3, d4);
  );

  // // Validate Result
  // size_t s0=0, s1=0, s2=0, s3=0, s4=0;
  // for(int i=0; i<h_A->get_rows(); i++){
  //   s0 += d0[i];
  //   s1 += d1[i];
  //   s2 += d2[i];
  //   s3 += d3[i];
  //   s4 += d4[i];
  // }
  // printf("s0:%lu\ns1:%lu\ns2:%lu\ns3:%lu\ns4:%lu\n", s0, s1, s2, s3, s4);



  // Free host memory
  free(d0);
  free(d1);
  free(d2);
  free(d3);
  free(d4);
}



int main(int argc, char *argv[]) {

  // Initialize cuda context

  // struct timespec T_START, T_END;
  std::chrono::high_resolution_clock::time_point T_START, T_END;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
    exit(1);
  }

  // Read mtx file
  h_csr h_A = loadSymmFileToCsr(argv[1]);

  std::cout << "0\t";
  TIME_OP("FGLT",   
    fglt(&h_A);
  );
  std::cout << "0\t" << std::endl;

  return 0;
}