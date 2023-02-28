#pragma once

#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "host_structs.hpp"
#include "device_csr.hpp"

template <typename T>
std::ostream &operator<<(std::ostream &os, const thrust::host_vector<T> &vec)
{
    if (vec.size() > 1000)
    {
        os << "Vector too large to print" << std::endl;
        return os;
    }
    os << "|";
    for (const T &el : vec)
    {
        os << " " << el << " |";
    }
    os << std::endl;
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const thrust::device_vector<T> &vec)
{
    if (vec.size() > 1000)
    {
        os << "Vector too large to print" << std::endl;
        return os;
    }
    os << "|";
    for (const T &el : vec)
    {
        os << " " << el << " |";
    }
    os << std::endl;
    return os;
}

template <typename T>
T *csrToRowMajor(const thrust::host_vector<int> &offsetsCSR, const thrust::host_vector<int> &columnsCSR, const thrust::host_vector<T> &valuesCSR, int rows, int cols, int nnz)
{
    T *rowMajor = new T[rows * cols];

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

template <typename T>
void printCSR(const thrust::host_vector<int> &offsetsCSR, const thrust::host_vector<int> &columnsCSR, const thrust::host_vector<T> &valuesCSR, int rows, int cols, int nnz)
{
    if (rows + cols > 1000)
    {
        std::cout << "Too big to print" << std::endl;
        return;
    }

    T *rowMajor = csrToRowMajor(offsetsCSR, columnsCSR, valuesCSR, rows, cols, nnz);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << std::setw(3) << rowMajor[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] rowMajor;
}

template <typename T>
T *csrToRowMajor(const thrust::device_vector<int> &offsetsCSR, const thrust::device_vector<int> &columnsCSR, const thrust::device_vector<T> &valuesCSR, int rows, int cols, int nnz)
{
    T *rowMajor = new T[rows * cols];

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

template <typename T>
void printCSR(const thrust::device_vector<int> &offsetsCSR, const thrust::device_vector<int> &columnsCSR, const thrust::device_vector<T> &valuesCSR, int rows, int cols, int nnz)
{
    if (rows * cols > 1000)
    {
        std::cout << "Too big to print" << std::endl;
        return;
    }

    T *rowMajor = csrToRowMajor(offsetsCSR, columnsCSR, valuesCSR, rows, cols, nnz);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << std::setw(3) << rowMajor[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] rowMajor;
}

// special case for h_csr and d_csr
void printCSR(const h_csr &csr);
void printCSR(const d_csr &csr);