#include <thrust/device_vector.h>

#include "device_csr.hpp"
#include "printing.hpp"

d_csr::d_csr() : d_csr(0, 0, 0) {}

d_csr::d_csr(int rows, int cols, int nnz)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets = thrust::device_vector<int>(rows + 1);
    this->positions = thrust::device_vector<int>(nnz);
    this->values = thrust::device_vector<COMPUTE_TYPE>(nnz);
}

d_csr::d_csr(int rows, int cols, int nnz, thrust::device_vector<int> &offsets, thrust::device_vector<int> &positions, thrust::device_vector<COMPUTE_TYPE> &values)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets = offsets;
    this->positions = positions;
    this->values = values;
}

d_csr::d_csr(int rows, int cols, int nnz, thrust::device_vector<int> &&offsets, thrust::device_vector<int> &&positions, thrust::device_vector<COMPUTE_TYPE> &&values)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets = std::move(offsets);
    this->positions = std::move(positions);
    this->values = std::move(values);
}

// move constructor
d_csr::d_csr(d_csr &&other) : d_csr(other.rows, other.cols, other.nnz, std::move(other.offsets), std::move(other.positions), std::move(other.values)) {}

// conversion constructor
d_csr::d_csr(const h_csr &_h_csr, bool copy_values) : rows(_h_csr.get_rows()), cols(_h_csr.get_cols()), nnz(_h_csr.get_nnz()),
                                                      offsets(_h_csr.get_offsets()), positions(_h_csr.get_positions())
{
    if (copy_values)
    {
        values = _h_csr.values;
    }
}

// getters
int d_csr::get_rows() const
{
    return rows;
}

int d_csr::get_cols() const
{
    return cols;
}

int d_csr::get_nnz() const
{
    return nnz;
}

const thrust::device_vector<int> &d_csr::get_offsets() const
{
    return offsets;
}

const thrust::device_vector<int> &d_csr::get_positions() const
{
    return positions;
}

const thrust::device_vector<COMPUTE_TYPE> &d_csr::get_values() const
{
    return values;
}

void d_csr::resize(int rows, int cols, int nnz)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets.resize(rows + 1);
    this->positions.resize(nnz);
    this->values.resize(nnz);
}

thrust::device_vector<COMPUTE_TYPE> d_csr::spmv_symbolic(const d_csr &A, const thrust::device_vector<COMPUTE_TYPE> &x)
{

    const int n = A.rows;
    thrust::device_vector<COMPUTE_TYPE> y(n);

    const int SmSize = 16;
    const int threadsPerBlock = 256;
    const int threadsPerSM = threadsPerBlock * SmSize;

    const int FullSMs = (n + threadsPerSM - 1) / threadsPerSM;

    spmv_symbolic_kernel<<<FullSMs * SmSize, threadsPerBlock>>>(
        A.offsets.data().get(), A.positions.data().get(), x.data().get(), y.data().get(), n);

    cudaDeviceSynchronize();

    return y;
}

__global__ void spmv_symbolic_kernel(const int *A_offsets, const int *A_positions, const COMPUTE_TYPE *x, COMPUTE_TYPE *y, const int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        COMPUTE_TYPE sum = 0;
        for (int j = A_offsets[i]; j < A_offsets[i + 1]; j++)
        {
            sum += x[A_positions[j]];
        }
        y[i] = sum;
    }
}