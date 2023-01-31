#include <cuda.h>
#include <cusparse.h>
#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/transform.h>

#include <cstddef>

#include "common/fileio.hpp"
#include "common/sparse_funcs.hpp"
#include "common/printing.hpp"

#define CHECK_CUDA(call)                                               \
    {                                                                  \
        cudaError_t status = (call);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            exit(1);                                                   \
        }                                                              \
    }

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

template<typename T>
__global__ void elwiseMul(T* a, T* b, T* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

#define BLOCK_SIZE 256
template<typename T>
__global__ void adjDif(T* in, T* out, int n){
    __shared__ T temp[BLOCK_SIZE + 1];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x;

    if(gindex < n){
        // Load input into shared memory
        temp[lindex] = in[gindex];
        if(lindex == BLOCK_SIZE-1 || gindex == n-1){
            temp[lindex+1] = in[gindex + 1];
        }
    }

    __syncthreads();

    // Compute the difference
    if(gindex < n)
        out[gindex] = temp[lindex + 1] - temp[lindex];
}

template<typename T = void>
struct d3_trans
{
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr T operator()(const T &lhs, const T &rhs) const
  {
    return (lhs)*(lhs-1) - 2*rhs;
  }
};

int main(int argc, char *argv[])
{

    // HOST DATA
    Symm_Sparse_matrix h_sA = loadFileToSymmSparse("/cuda_project/datasets/test.mtx");
    thrust::host_vector<float> h_sA_vals(h_sA.nnz, 1.0f);

    const int n = h_sA.n;
    const int nnz = h_sA.nnz;

    int A_rows = n;
    int A_cols = n;
    int A_nnz = nnz;

    std::cout << "A = " << std::endl;
    printCSR(h_sA.offsets, h_sA.positions, h_sA_vals, n, n, nnz);
    std::cout << "A Sparsity: " << (100 * ( 1 - ((float)h_sA.nnz / (float)(n * n)))) << "%" << std::endl;
    std::cout << "A positions: " << h_sA.positions.size() << std::endl;

    // DEVICE DATA
    const int64_t A2_rows = A_rows;
    const int64_t A2_cols = A_cols;
    int64_t A2_nnz = 0;

    // DEVICE DATA
    thrust::device_vector<int> d_A_offs(h_sA.offsets);
    thrust::device_vector<int> d_A_cols(h_sA.positions);
    thrust::device_vector<float> d_A_vals(h_sA_vals);

    thrust::device_vector<int> d_A2_offs(A2_rows + 1);
    thrust::device_vector<int> d_A2_cols;
    thrust::device_vector<float> d_A2_vals;

    // CUSPARSE STBRTS HERE
    const cudaDataType computeType = CUDA_R_32F;

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))

/*

        CSR MATRIX CREATION

*/

    cusparseSpMatDescr_t A_CSR;
    CHECK_CUSPARSE(
        cusparseCreateCsr(
            &A_CSR,
            A_rows,
            A_cols,
            A_nnz,
            d_A_offs.data().get(),
            d_A_cols.data().get(),
            d_A_vals.data().get(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            computeType
        )
    )

    cusparseSpMatDescr_t A2_CSR;

    CHECK_CUSPARSE(
        cusparseCreateCsr(
            &A2_CSR,
            A2_rows,
            A2_cols,
            0,
            NULL,
            NULL,
            NULL,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            computeType
        )
    )

/*

            A_2 = A*A COMPUTATION

*/

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

    // void *buffer1_d = NULL, *buffer2_d = NULL;
    //  in thrust dialect:

    float alpha = 1.0f, beta = 0.0f;
    const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    size_t bufferSize1 = 0;
// estimate memmory needed for this
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(
            handle,
            opA,
            opB,
            &alpha,
            A_CSR,
            A_CSR,
            &beta,
            A2_CSR,
            computeType,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize1,
            NULL
        )
    )
    std::cout << "bufferSize1_d: " << bufferSize1 << std::endl;
    
    thrust::device_vector<uint8_t> d_buffer1(bufferSize1);
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(
            handle,
            opA,
            opB,
            &alpha,
            A_CSR,
            A_CSR,
            &beta,
            A2_CSR,
            computeType,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize1,
            d_buffer1.data().get()
        )
    )

    size_t bufferSize2 = 0;
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(
            handle,
            opA,
            opB,
            &alpha,
            A_CSR,
            A_CSR,
            &beta,
            A2_CSR,
            computeType,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize2,
            NULL
        )
    )
    std::cout << "bufferSize2_d: " << bufferSize2 << std::endl;
    thrust::device_vector<uint8_t> d_buffer2(bufferSize2);

    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(
            handle,
            opA,
            opB,
            &alpha,
            A_CSR,
            A_CSR,
            &beta,
            A2_CSR,
            computeType,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc,
            &bufferSize2,
            d_buffer2.data().get()
        )
    )

    int64_t throwaway;
    CHECK_CUSPARSE(cusparseSpMatGetSize(A2_CSR, &throwaway, &throwaway, &A2_nnz));

    std::cout << "A2_rows: " << A2_rows << std::endl;
    std::cout << "A2_cols: " << A2_cols << std::endl;
    std::cout << "A2_nnz: " << A2_nnz << std::endl;

    d_A2_offs.resize(A2_rows + 1);
    d_A2_vals.resize(A2_nnz);
    d_A2_cols.resize(A2_nnz);

    CHECK_CUSPARSE(
        cusparseCsrSetPointers(
            A2_CSR,
            d_A2_offs.data().get(),
            d_A2_cols.data().get(),
            d_A2_vals.data().get()
        )
    )

    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(
            handle,
            opA,
            opB,
            &alpha,
            A_CSR,
            A_CSR,
            &beta,
            A2_CSR,
            computeType,
            CUSPARSE_SPGEMM_DEFAULT,
            spgemmDesc
        )
    )

    std::cout << "A2 CSR matrix created" << std::endl;
    std::cout << "A2 Sparsity: " << (1 - ((float)A2_nnz / (A2_rows * A2_cols))) * 100 << "%" << std::endl;
    std::cout << "A2_offs: " << d_A2_offs;
    printCSR(d_A2_offs, d_A2_cols, d_A2_vals, A2_rows, A2_cols, A2_nnz);
//DONE

/*

            CALCULATE p1

*/
    thrust::device_vector<float> d_p1(n);
    // make a copy of d_A_offs but with floats
    thrust::device_vector<float> d_A_offs_float(d_A_offs);

    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adjDif<<<num_blocks, BLOCK_SIZE>>>(d_A_offs_float.data().get(), d_p1.data().get(), n);
    std::cout << "p1: " << d_p1;

// DONE 
/*

        HADAMARD PRODUCT OF A2 * A

*/

                // Copy A2 to host
    thrust::host_vector<int> h_A2_offs(d_A2_offs);
    thrust::host_vector<int> h_A2_cols(d_A2_cols);
    thrust::host_vector<float> h_A2_vals(d_A2_vals);


                // Create space for hadamard product
    thrust::host_vector<int> h_C3_offs(A2_rows+1);
    thrust::host_vector<int> h_C3_cols(A_nnz + A2_nnz);
    thrust::host_vector<float> h_C3_vals(A_nnz + A2_nnz);

                // Calculate hadamard product
    int had_nnz = csr_hadmul_csr_canonical(
        A_rows,
        A_cols,
        h_sA.offsets.data(),
        h_sA.positions.data(),
        h_sA_vals.data(),
        h_A2_offs.data(),
        h_A2_cols.data(),
        h_A2_vals.data(),
        h_C3_offs.data(),
        h_C3_cols.data(),
        h_C3_vals.data()
    );

    h_C3_cols.resize(had_nnz);
    h_C3_vals.resize(had_nnz);

    std::cout << "A hadamard A2 = " << std::endl;
    printCSR(h_C3_offs, h_C3_cols, h_C3_vals, A2_rows, A2_cols, had_nnz);

    thrust::device_vector<int> d_C3_offs(h_C3_offs);
    thrust::device_vector<int> d_C3_cols(h_C3_cols);
    thrust::device_vector<float> d_C3_vals(h_C3_vals);

    cusparseSpMatDescr_t C3_CSR;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &C3_CSR,
        A_rows,
        A_cols,
        had_nnz,
        d_C3_offs.data().get(),
        d_C3_cols.data().get(),
        d_C3_vals.data().get(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ))


// DONE

/*

            CALCULATE c3

*/
    thrust::device_vector<float> d_e(n, 1);
    thrust::device_vector<float> d_c3(n); 

    cusparseDnVecDescr_t e_descr;
    cusparseDnVecDescr_t c3_descr;

    CHECK_CUSPARSE(cusparseCreateDnVec(&e_descr, n, d_e.data().get(), CUDA_R_32F))

    CHECK_CUSPARSE(cusparseCreateDnVec(&c3_descr, n, d_c3.data().get(), CUDA_R_32F))

    size_t bufferSizeC3_1 = 0;

    float alpha_c3 = 0.5;

    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_c3,
            C3_CSR,
            e_descr,
            &beta,
            c3_descr,
            computeType,
            CUSPARSE_MV_ALG_DEFAULT,
            &bufferSizeC3_1
        )
    )

    thrust::device_vector<uint8_t> d_bufferC3(bufferSizeC3_1);

    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_c3,
        C3_CSR,
        e_descr,
        &beta,
        c3_descr,
        computeType,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_bufferC3.data().get()
        )
    )

    std::cout << "c3: " << d_c3;

/*

            CALCULATE p2

*/
    thrust::device_vector<float> d_Ap1(n);

    cusparseDnVecDescr_t p1_descr;
    cusparseDnVecDescr_t Ap1_descr;

    CHECK_CUSPARSE(cusparseCreateDnVec(&p1_descr, n, d_p1.data().get(), CUDA_R_32F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&Ap1_descr, n, d_Ap1.data().get(), CUDA_R_32F))

    size_t bufferSizeAp1 = 0;

    float alpha_Ap1 = 1.0;

    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_Ap1,
            A_CSR,
            p1_descr,
            &beta,
            Ap1_descr,
            computeType,
            CUSPARSE_MV_ALG_DEFAULT,
            &bufferSizeAp1
        )
    )

    thrust::device_vector<int> d_bufferAp1(bufferSizeAp1);

    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_Ap1,
        A_CSR,
        p1_descr,
        &beta,
        Ap1_descr,
        computeType,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_bufferAp1.data().get()
        )
    )

    thrust::device_vector<float> d_p2(n);

    thrust::transform(
        d_Ap1.begin(),
        d_Ap1.end(),
        d_p1.begin(),
        d_p2.begin(),
        thrust::minus<float>()
    );

/*

            CALCULATE d2

*/

    thrust::device_vector<float> d_d2(n);

    thrust::transform(
        d_p2.begin(),
        d_p2.end(),
        d_c3.begin(),
        d_d2.begin(),
        thrust::minus<float>()
    );

/*

            CALCULATE d3

*/

    thrust::device_vector<float> d_d3(n);

    thrust::transform(
        d_p1.begin(),
        d_p1.end(),
        d_c3.begin(),
        d_d3.begin(),
        d3_trans<float>()
    );

// DONE

    std::cout << "d1 = " << d_p1;
    std::cout << "d2 = " << d_d2;
    std::cout << "d3 = " << d_d3;
    std::cout << "d4 = " << d_c3;
    return 0;
}