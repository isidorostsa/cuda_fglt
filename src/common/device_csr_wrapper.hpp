#pragma once

#include <thrust/device_vector.h>
#include <cusparse.h>

#include <thrust/host_vector.h>

#include "host_structs.hpp"

class h_csr;

class d_cusparse_csr
{
private:
    cusparseSpMatDescr_t desc = nullptr;

    size_t rows, cols, nnz;

    thrust::device_vector<int> offsets;
    thrust::device_vector<int> positions;
    thrust::device_vector<float> values;

    const cudaDataType_t compute_type = CUDA_R_32F;


    void update_descriptor();

public:
    d_cusparse_csr();

    d_cusparse_csr(size_t rows, size_t cols);

    d_cusparse_csr(size_t rows, size_t cols, size_t nnz);

    d_cusparse_csr(d_cusparse_csr &&other);

    d_cusparse_csr(size_t rows, size_t cols, size_t nnz, const thrust::device_vector<int> &offsets, const thrust::device_vector<int> &positions, const thrust::device_vector<float> &values);

    d_cusparse_csr(size_t rows, size_t cols, size_t nnz, const thrust::host_vector<int> &offsets, const thrust::host_vector<int> &positions, const thrust::host_vector<float> &values);

    d_cusparse_csr(size_t rows, size_t cols, size_t nnz, thrust::device_vector<int> &&offsets, thrust::device_vector<int> &&positions, thrust::device_vector<float> &&values);

    d_cusparse_csr(const h_csr &h_csr);

    ~d_cusparse_csr();

    void resize_vectors(size_t offsets_size, size_t position_value_size);

    void update_vectors();

    void update_descriptor_pointers();

    void follow_descriptor();

    void take(size_t rows, size_t cols, size_t nnz, thrust::device_vector<int> &&offsets, thrust::device_vector<int> &&positions, thrust::device_vector<float> &&values);

    // move constructor

    static d_cusparse_csr multiply(const d_cusparse_csr &A, const d_cusparse_csr &B, cusparseHandle_t cusparseHandle);

    static thrust::device_vector<float> multiply(const d_cusparse_csr &A, const thrust::device_vector<float>& v, cusparseHandle_t cusparseHandle, float alpha = 1.0f, float beta = 0.0f);

    int getRows() const;
    int getCols() const;
    int getNnz() const;

    const thrust::device_vector<int> &get_offsets() const;

    const thrust::device_vector<int> &get_positions() const;

    const thrust::device_vector<float> &get_values() const;

    const cusparseSpMatDescr_t& get_descriptor() const;

    // cool but not used operator h_csr() const;
};