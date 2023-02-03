#pragma once

#include <iostream>
#include <vector>

#include "host_structs.hpp"

Coo_matrix loadFileToCoo(const std::string filename);

h_csr loadFileToCsr(const std::string filename);

void coo_tocsr(const Coo_matrix& coo, h_symm_csr& csr);

void coo_tocsc(const Coo_matrix& coo, h_symm_csr& csx);

void csr_tocsc(const int n, const thrust::host_vector<int>& Ap, const thrust::host_vector<int>& Aj, 
	                thrust::host_vector<int>& Bp, thrust::host_vector<int>& Bi);