#pragma once

#include <thrust/device_vector.h>
#include "host_structs.hpp"

class h_csr;

#define COMPUTE_TYPE int

struct d_csr {
    int rows;
    int cols;
    int nnz;

    thrust::device_vector<int> offsets;
    thrust::device_vector<int> positions;
    thrust::device_vector<COMPUTE_TYPE> values;

    d_csr();
    d_csr(int rows, int cols, int nnz);
    d_csr(int rows, int cols, int nnz, thrust::device_vector<int>& offsets, thrust::device_vector<int>& positions, thrust::device_vector<COMPUTE_TYPE>& values);
    d_csr(int rows, int cols, int nnz, thrust::device_vector<int>&& offsets, thrust::device_vector<int>&& positions, thrust::device_vector<COMPUTE_TYPE>&& values);
    d_csr(d_csr &&other);
    d_csr(const h_csr &_h_csr);

    // getters and setters
    int get_rows() const;
    int get_cols() const;
    int get_nnz() const;

    const thrust::device_vector<int>& get_offsets() const;
    const thrust::device_vector<int>& get_positions() const;
    const thrust::device_vector<COMPUTE_TYPE>& get_values() const;

    void resize(int rows, int cols, int nnz);

    static thrust::device_vector<COMPUTE_TYPE> spmv_symbolic(const d_csr &A, const thrust::device_vector<COMPUTE_TYPE> &x);
};

__global__ void spmv_symbolic_kernel(const int* A_offsets, const int* A_positions, const COMPUTE_TYPE* x, COMPUTE_TYPE* y, const int n);