#include <iostream>
#include <cassert>

#include <cusparse.h>
#include <thrust/device_vector.h>

#include "cusparse_csr_wrapper.hpp"

#define CHECK_CUSPARSE(call)                                                                           \
    {                                                                                                  \
        cusparseStatus_t status = call;                                                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                         \
        {                                                                                              \
            fprintf(stderr, "cuSparse error %s in file '%s' in line %i : %s.\n",                       \
                    cusparseGetErrorName(status), __FILE__, __LINE__, cusparseGetErrorString(status)); \
            exit(1);                                                                                   \
        }                                                                                              \
    }

d_cusparse_csr::d_cusparse_csr() : d_cusparse_csr(0, 0, 0){};

d_cusparse_csr::d_cusparse_csr(size_t rows, size_t cols) : d_cusparse_csr(rows, cols, 0){};

d_cusparse_csr::d_cusparse_csr(size_t rows, size_t cols, size_t nnz) : rows(rows), cols(cols), nnz(nnz)
{

    offsets.resize(rows + 1);
    positions.resize(nnz);
    values.resize(nnz);

    update_descriptor();
}

d_cusparse_csr::d_cusparse_csr(size_t rows, size_t cols, size_t nnz, const thrust::device_vector<int> &offsets, const thrust::device_vector<int> &positions, const thrust::device_vector<float> &values)
    : rows(rows), cols(cols), nnz(nnz), offsets(offsets), positions(positions), values(values)
{
    update_descriptor();
}

d_cusparse_csr::d_cusparse_csr(size_t rows, size_t cols, size_t nnz, thrust::device_vector<int> &&offsets, thrust::device_vector<int> &&positions, thrust::device_vector<float> &&values)
    : rows(rows), cols(cols), nnz(nnz), offsets(std::move(offsets)), positions(std::move(positions)), values(std::move(values))
{
    update_descriptor();
}

d_cusparse_csr::d_cusparse_csr(size_t rows, size_t cols, size_t nnz, const thrust::host_vector<int> &values, const thrust::host_vector<int> &offsets, const thrust::host_vector<float> &positions)
    : rows(rows), cols(cols), nnz(nnz), offsets(offsets), positions(positions), values(values)
{
    update_descriptor();
}

d_cusparse_csr::~d_cusparse_csr()
{
    CHECK_CUSPARSE(cusparseDestroySpMat(desc));
}

void d_cusparse_csr::take(size_t rows, size_t cols, size_t nnz, thrust::device_vector<int> &&offsets, thrust::device_vector<int> &&positions, thrust::device_vector<float> &&values)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;

    offsets = std::move(offsets);
    positions = std::move(positions);
    values = std::move(values);

    update_descriptor();
}

void d_cusparse_csr::update_vectors()
{
    offsets.resize(rows + 1);
    positions.resize(nnz);
    values.resize(nnz);

    update_descriptor();
}

void d_cusparse_csr::update_descriptor()
{
    assert(offsets.size() == rows + 1);
    assert(positions.size() == nnz);
    assert(values.size() == nnz);

    if (nnz == 0 || rows == 0 || cols == 0)
    {
        CHECK_CUSPARSE(
            cusparseCreateCsr(
                &desc,
                rows,
                cols,
                0,
                NULL,
                NULL,
                NULL,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                compute_type))
        return;
    }

    int64_t rows64 = static_cast<int64_t>(rows);
    int64_t cols64 = static_cast<int64_t>(cols);
    int64_t nnz64 = static_cast<int64_t>(nnz);

    CHECK_CUSPARSE(
        cusparseCreateCsr(
            &desc,
            rows64,
            cols64,
            nnz64,
            offsets.data().get(),
            positions.data().get(),
            values.data().get(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            compute_type))
}

d_cusparse_csr::d_cusparse_csr(d_cusparse_csr &&other) : rows(other.rows), cols(other.cols), nnz(other.nnz),
                                                         offsets(std::move(other.offsets)), positions(std::move(other.positions)), values(std::move(other.values))

{
    update_descriptor();
}

d_cusparse_csr::d_cusparse_csr(const h_csr &h_csr_) : rows(h_csr_.rows), cols(h_csr_.cols), nnz(h_csr_.nnz),
                                                      offsets(h_csr_.offsets), positions(h_csr_.positions), values(h_csr_.values)
{
    update_descriptor();
}

// MATIX MULTIPLICATION
d_cusparse_csr d_cusparse_csr::multiply(const d_cusparse_csr &A, const d_cusparse_csr &B, cusparseHandle_t handle)
{

    assert(A.cols == B.rows);
    assert(A.compute_type == B.compute_type);

    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    d_cusparse_csr C(A.rows, B.cols);

    const auto compute_type = C.compute_type;

    float alpha = 1.0f;
    float beta = 0.0f;

    // inspect SpGEMM work space

    size_t bufferSize1;
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(
            handle,
            opA,
            opB,
            &alpha,
            A.get_descriptor(),
            B.get_descriptor(),
            &beta,
            C.get_descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize1,
            NULL))

    std::cout << "Megabytes needed for SpGEMM 1: " << bufferSize1 / 1024.0 / 1024.0 << " MB" << std::endl;

    // thrust::device_vector<uint8_t> d_buffer1(bufferSize1*2);

    void* d_buffer1;

    cudaMalloc((void **)&d_buffer1, bufferSize1);

    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(
            handle,
            opA,
            opB,
            &alpha,
            A.get_descriptor(),
            B.get_descriptor(),
            &beta,
            C.get_descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize1,
            // d_buffer1.data().get()
            d_buffer1
        )
    )

    size_t bufferSize2 = 0;
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(
            handle,
            opA,
            opB,
            &alpha,
            A.get_descriptor(),
            B.get_descriptor(),
            &beta,
            C.get_descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize2,
            NULL
        )
    )

    std::cout << "Megabytes needed for SpGEMM 2: " << bufferSize2 / 1024.0 / 1024.0 << " MB" << std::endl;

    thrust::device_vector<char> d_buffer2(bufferSize2);

    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(
            handle,
            opA,
            opB,
            &alpha,
            A.get_descriptor(),
            B.get_descriptor(),
            &beta,
            C.get_descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize2,
            d_buffer2.data().get()))

    // now in buffer2 resides the result of the multiplication

    // !!!!!!!!!!!!!!!!!!!!!!!!
    /// IMPORTANT PART HERE!!!!
    // !!!!!!!!!!!!!!!!!!!!!!!!
    C.follow_descriptor();

    // we need to copy the result into C, but we need to resize C's vectors first
    // this will repoint the descriptor's pointers to the new empty vectors

    // now we can copy the result into the new C descriptor
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(
            handle,
            opA,
            opB,
            &alpha,
            A.get_descriptor(),
            B.get_descriptor(),
            &beta,
            C.get_descriptor(),
            compute_type,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc))

    // Now c is in the correct state, with the values from the descriptor copied into the vectors

    cusparseSpGEMM_destroyDescr(spgemmDesc);

    return C;
}

// MATRIX-VECTOR MULTIPLICATION

thrust::device_vector<float> d_cusparse_csr::multiply(const d_cusparse_csr &A, const thrust::device_vector<float> &v, cusparseHandle_t handle, float alpha, float beta)
{
    const size_t n = v.size();

    thrust::device_vector<float> v_out(v.size());

#if CUSPARSE_VERSION >= 12000
    cusparseConstDnVecDescr_t v_desc;
    CHECK_CUSPARSE(cusparseCreateConstDnVec(&v_desc, v.size(), static_cast<const void *>(v.data().get()), A.compute_type))
#else
    cusparseDnVecDescr_t v_desc;
    CHECK_CUSPARSE(cusparseCreateDnVec(&v_desc, v.size(), const_cast<void *>(static_cast<const void *>(v.data().get())), A.compute_type))
#endif
    cusparseDnVecDescr_t v_out_desc;
    CHECK_CUSPARSE(cusparseCreateDnVec(&v_out_desc, n, static_cast<void *>(v_out.data().get()), A.compute_type))

    size_t bufferSize = 0;

    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            A.get_descriptor(),
            v_desc,
            &beta,
            v_out_desc,
            A.compute_type,
#if CUSPARSE_VERSION >= 12000
            CUSPARSE_SPMV_ALG_DEFAULT,
#else
            CUSPARSE_SPMV_ALG1,
#endif
            &bufferSize))

    thrust::device_vector<char> d_buffer(bufferSize);

    CHECK_CUSPARSE(
        cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            A.get_descriptor(),
            v_desc,
            &beta,
            v_out_desc,
            A.compute_type,
#if CUSPARSE_VERSION >= 12000
            CUSPARSE_SPMV_ALG_DEFAULT,
#else
            CUSPARSE_SPMV_ALG1,
#endif
            static_cast<void *>(d_buffer.data().get())))

    cusparseDestroyDnVec(v_desc);

    return v_out;
}

int d_cusparse_csr::get_rows() const { return rows; }
int d_cusparse_csr::get_cols() const { return cols; }
int d_cusparse_csr::get_nnz() const { return nnz; }

const thrust::device_vector<int> &d_cusparse_csr::get_offsets() const
{
    return offsets;
}

const thrust::device_vector<int> &d_cusparse_csr::get_positions() const
{
    return positions;
}

const thrust::device_vector<float> &d_cusparse_csr::get_values() const
{
    return values;
}

const cusparseSpMatDescr_t &d_cusparse_csr::get_descriptor() const
{
    return desc;
}

void d_cusparse_csr::follow_descriptor()
{
    int64_t rows_desc, cols_desc, nnz_desc;

    CHECK_CUSPARSE(cusparseSpMatGetSize(desc, &rows_desc, &cols_desc, &nnz_desc));

    this->rows = static_cast<size_t>(rows_desc);
    this->cols = static_cast<size_t>(cols_desc);
    this->nnz = static_cast<size_t>(nnz_desc);

    offsets.resize(rows + 1);
    positions.resize(nnz);
    values.resize(nnz);

    update_descriptor_pointers();
}

void d_cusparse_csr::resize_vectors(size_t offset_size, size_t position_value_size)
{
    this->rows = offset_size - 1;
    this->cols = position_value_size;
    this->nnz = position_value_size;

    offsets.resize(offset_size);
    positions.resize(position_value_size);
    values.resize(position_value_size);

    update_descriptor();
}

void d_cusparse_csr::update_descriptor_pointers()
{
    CHECK_CUSPARSE(cusparseCsrSetPointers(desc, offsets.data().get(), positions.data().get(), values.data().get()))
}

// a kernel that first computes the number of nonzeros in each block
// and then copies the values into the correct positions
// __global__ void hadamard_kernel(const int n, const int A_nnz, const int B_nnz, const int *A_offsets, const int *A_positions, const float *A_values, const int *B_offsets, const int *B_positions, const float *B_values, int *C_offsets, int *C_positions, float *C_values)
// {
//     int block = blockIdx.x;
//     int tid = threadIdx.x;

//     int els_per_block = (n + gridDim.x - 1) / gridDim.x;

//     int block_start = block * els_per_block;
//     int block_end = min((block + 1) * els_per_block, n);

//     int block_size = block_end - block_start;

// }


/* cool but not used
d_cusparse_csr::operator h_csr() const
{
    thrust::host_vector<int> h_offsets(offsets);
    thrust::host_vector<int> h_positions(positions);
    thrust::host_vector<float> h_values(values);

    return h_csr(rows, cols, nnz, std::move(h_offsets), std::move(h_positions), std::move(h_values));
}
*/