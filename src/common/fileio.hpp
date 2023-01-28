#pragma once

#include <iostream>
#include <vector>

#include <thrust/host_vector.h>

struct Coo_matrix {
    int n;
    int nnz;
    thrust::host_vector<int> Ai;
    thrust::host_vector<int> Aj;

    int real_nnz() { return 2*nnz; }
};

struct Symm_Sparse_matrix {
    int n;
    int nnz;
    thrust::host_vector<int> offsets;
    thrust::host_vector<int> positions;

    int real_nnz() { return 2*nnz; }
};

Coo_matrix loadFileToCoo(const std::string filename);

Symm_Sparse_matrix loadFileToSymmSparse(const std::string filename);

void coo_tocsr(const Coo_matrix& coo, Symm_Sparse_matrix& csr);

void coo_tocsc(const Coo_matrix& coo, Symm_Sparse_matrix& csc);

void csr_tocsc(const int n, const thrust::host_vector<int>& Ap, const thrust::host_vector<int>& Aj, 
	                thrust::host_vector<int>& Bp, thrust::host_vector<int>& Bi);

void csc_tocsr(const Symm_Sparse_matrix& csc, Symm_Sparse_matrix& csr);
void csr_tocsc(const Symm_Sparse_matrix& csr, Symm_Sparse_matrix& csc);