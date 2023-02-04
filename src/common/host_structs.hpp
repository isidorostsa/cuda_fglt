#pragma once

#include <thrust/host_vector.h>
#include "device_csr_wrapper.hpp"

class d_cusparse_csr;

struct Coo_matrix {
    int n;
    int nnz;
    thrust::host_vector<int> Ai;
    thrust::host_vector<int> Aj;
};

struct h_symm_csr {
    int n, nnz;

    thrust::host_vector<int> offsets;
    thrust::host_vector<int> positions;
    thrust::host_vector<float> values;

    int real_nnz();
};

struct h_csr {
    int rows;
    int cols;
    int nnz;

    thrust::host_vector<int> offsets;
    thrust::host_vector<int> positions;
    thrust::host_vector<float> values;

    h_csr();
    h_csr(int rows, int cols, int nnz);
    h_csr(int rows, int cols, int nnz, thrust::host_vector<int>& offsets, thrust::host_vector<int>& positions, thrust::host_vector<float>& values);
    h_csr(int rows, int cols, int nnz, thrust::host_vector<int>&& offsets, thrust::host_vector<int>&& positions, thrust::host_vector<float>&& values);
    h_csr(h_csr &&other);

    h_csr(const d_cusparse_csr &other);

    void resize(int rows, int cols, int nnz);

    static h_csr hadamard(const h_csr &A, const h_csr &B);

    // cool operator solution creates circular dependency: operator d_cusparse_csr() const;
};
