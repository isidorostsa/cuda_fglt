#pragma once

#include <iostream>
#include <vector>

#include "host_structs.hpp"

h_coo loadFileToCoo(const std::string& filename);

h_coo loadSymmFileToCoo(const std::string& filename);

h_csr loadFileToCsr(const std::string& filename);

h_csr loadSymmFileToCsr(const std::string& filename);

void coo_tocsr(const h_coo& coo, h_symm_csr& csr);

void coo_tocsc(const h_coo& coo, h_symm_csr& csx);

void csr_tocsc(const int n, const thrust::host_vector<int>& Ap, const thrust::host_vector<int>& Aj, 
	                thrust::host_vector<int>& Bp, thrust::host_vector<int>& Bi);

h_csr coo_to_csr(const h_coo &coo);