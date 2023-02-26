#include "printing.hpp"
#include "host_structs.hpp"
#include "device_csr.hpp"

void printCSR(const h_csr &csr)
{
    printCSR(csr.get_offsets(), csr.get_positions(), csr.get_values(), csr.get_rows(), csr.get_cols(), csr.get_nnz());
}

void printCSR(const d_csr &csr)
{
    printCSR(csr.get_offsets(), csr.get_positions(), csr.get_values(), csr.get_rows(), csr.get_cols(), csr.get_nnz());
}

