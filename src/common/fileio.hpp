#pragma once

#include <iostream>
#include <vector>

#include "host_structs.hpp"

h_coo loadSymmFileToCoo(const std::string& filename);

h_csr loadSymmFileToCsr(const std::string& filename);

h_csr coo_to_csr(const h_coo &coo);