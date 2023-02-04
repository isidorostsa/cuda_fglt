#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/printing.hpp"

int main(){
    int rows = 3;
    int cols = 3;
    int nnz = 5;

    int offsetsCSR[] = {0, 2, 3, 5};
    int columnsCSR[] = {0, 1, 2, 0, 2};
    float valuesCSR[] = {1, 2, 3, 4, 5};

    thrust::host_vector<int> offsetsCSR_h(offsetsCSR, offsetsCSR + rows + 1);
    thrust::host_vector<int> columnsCSR_h(columnsCSR, columnsCSR + nnz);
    thrust::host_vector<float> valuesCSR_h(valuesCSR, valuesCSR + nnz);

    printCSR(offsetsCSR_h, columnsCSR_h, valuesCSR_h, rows, cols, nnz);

    // now with device vectors

    thrust::device_vector<int> offsetsCSR_d(offsetsCSR, offsetsCSR + rows + 1);
    thrust::device_vector<int> columnsCSR_d(columnsCSR, columnsCSR + nnz);
    thrust::device_vector<float> valuesCSR_d(valuesCSR, valuesCSR + nnz);

    printCSR(offsetsCSR_d, columnsCSR_d, valuesCSR_d, rows, cols, nnz);

    return 0;
}
