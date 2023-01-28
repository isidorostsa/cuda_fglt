#include <cuda.h>
#include <cusparse.h>
#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstddef>

#include "common/fileio.hpp"
#include "common/sparse_funcs.hpp"

#define ln std::cout << __LINE__ << std::endl;

#define CHECK_CUDA(call)                                               \
    {                                                                  \
        cudaError_t status = (call);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
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

float *csrToRowMajor(int *columnsCSR, int *offsetsCSR, float *valuesCSR, int rows, int cols, int nnz)
{
    float *rowMajor = new float[rows * cols];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            rowMajor[i * cols + j] = 0;
        }
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = offsetsCSR[i]; j < offsetsCSR[i + 1]; j++)
        {
            rowMajor[i * cols + columnsCSR[j]] = valuesCSR[j];
        }
    }

    return rowMajor;
}

void printCSR(int *columnsCSR, int *offsetsCSR, float *valuesCSR, int rows, int cols, int nnz)
{
    float *rowMajor = csrToRowMajor(columnsCSR, offsetsCSR, valuesCSR, rows, cols, nnz);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << std::setw(5) << rowMajor[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const thrust::host_vector<T>& vec)
{
    os << "|";
    for (const T& el : vec) {
        os << " " << el << " |";
    } os << std::endl;
    return os;
}

int main(int argc, char *argv[])
{
    // HOST DATA
    Symm_Sparse_matrix h_sA = loadFileToSymmSparse("/cuda_project/datasets/auto/auto.mtx");
    thrust::host_vector<float> h_sA_vals(h_sA.nnz, 1.0f);

    std::cout << "A = " << std::endl;
    //printCSR(h_sA.positions.data(), h_sA.offsets.data(), h_sA_vals.data(), h_sA.n, h_sA.n, h_sA.n);

    int A_rows = h_sA.n;
    int A_cols = h_sA.n;
    int A_nnz = h_sA.nnz;

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
    const cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cudaDataType computeType = CUDA_R_32F;

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    //CHECK_CUSPARSE(cusparseLoggerSetMask(1 | 0 | 4 | 8 | 16))

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
            computeType))

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

    size_t bufferSize1 = 0;
    // estimate memmory needed for this
    // PASSING NULL AS THE LAST PARAMETER
    // TELLS CUSPARSE TO HANDLE MEMMORY IN ITS OWN BUFFER
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
    
    thrust::device_vector<uint8_t> d_buffer1(bufferSize1);

    std::cout
        << "bufferSize1_d: " << bufferSize1 << std::endl;

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
            NULL))
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
//DONE

/*

            INITIALIZE VECTORS FOR SPMV

*/
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Now we will calculate A2 * [1 1 .... 1]^T
    thrust::host_vector<float> h_mulVec(A2_rows, 1.0f);
    thrust::device_vector<float> d_mulVec(h_mulVec);

    cusparseDnVecDescr_t mulVec_descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&mulVec_descr, A_rows, d_mulVec.data().get(), CUDA_R_32F))

    thrust::host_vector<float> h_resVec(A2_rows);
    thrust::device_vector<float> d_resVec(A2_rows);
    cusparseDnVecDescr_t resVec_descr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&resVec_descr, A_rows, d_resVec.data().get(), CUDA_R_32F))

/*

            PERFORM SPMV

*/

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        A2_CSR,
        mulVec_descr,
        &beta,
        resVec_descr,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
        )
    )

    thrust::device_vector<uint8_t> d_buffer(bufferSize);

    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        A2_CSR,
        mulVec_descr,
        &beta,
        resVec_descr,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer.data().get()
        )
    )

// DONE

/*

        HADAMARD PRODUCT OF A2 * A

*/

                // Copy A2 to host
    thrust::host_vector<int> h_A2_offs(d_A2_offs);
    thrust::host_vector<int> h_A2_cols(d_A2_cols);
    thrust::host_vector<float> h_A2_vals(d_A2_vals);


                // Create space for hadamard product
    thrust::host_vector<int> h_had_offs(A2_rows+1);
    thrust::host_vector<int> h_had_cols(A_nnz + A2_nnz);
    thrust::host_vector<float> h_had_vals(A_nnz + A2_nnz);

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
        h_had_offs.data(),
        h_had_cols.data(),
        h_had_vals.data()
    );

    h_had_cols.resize(had_nnz);
    h_had_vals.resize(had_nnz);

    std::cout << "A hadamard A2 = " << std::endl;
    //printCSR(h_had_cols.data(), h_had_offs.data(), h_had_vals.data(), A2_rows, A2_cols, had_nnz);

    CHECK_CUSPARSE(cusparseDestroySpMat(A_CSR));
    CHECK_CUSPARSE(cusparseDestroySpMat(A2_CSR));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    auto sparsity = [](thrust::host_vector<float>& vec) -> float
    {
        int nnz = 0;
        for (auto& val : vec)
        {
            if (std::abs(val) > 1e-10) 
            {
                nnz++;
            } 
        }
        return 1 - (float)nnz / (float)vec.size();
    };

    std::cout << "A2 = " << std::endl;
    //printCSR(h_A2_cols.data(), h_A2_offs.data(), h_A2_vals.data(), A_rows, A2_cols, A2_nnz);

    std::cout << "A2 * [1 1 ... 1]^T = ";

    h_resVec.resize(A2_rows);
    thrust::copy(d_resVec.begin(), d_resVec.end(), h_resVec.begin());
    for (int i = 0; i < A_rows; i++)
    {
        std::cout << h_resVec[i] << ", ";
    } std::cout << std::endl;

    std::cout << "Resulting vector sparsity: " << sparsity(h_resVec)*100 << "%" << std::endl;

    return 0;
}