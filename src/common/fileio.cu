#include <iostream>
#include <string>
#include <fstream>

#include <thrust/host_vector.h>

#include "fileio.hpp"
#include "host_structs.hpp"

#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wunused-result"

h_coo loadSymmFileToCoo(const std::string& filename)
{
    std::ifstream fin(filename);
    // check if file exists
    if(!std::ifstream(filename).good()) {
        std::cout << "File " << filename << " does not exist" << std::endl;
        exit(1);
    }


    int n, nnz;
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');

    fin >> n >> n >> nnz;

    thrust::host_vector<int> Ai(2*nnz);
    thrust::host_vector<int> Aj(2*nnz);

    int throwaway;
    // lines may be of the form: i j or i j throwaway
    for (int i = 0; i < nnz; i++)
    {
        int array_index = 2*i;
        fin >> Ai[array_index] >> Aj[array_index];
        Ai[array_index]--;
        Aj[array_index]--;

        if (Ai[array_index] == Aj[array_index]){
            throw std::runtime_error("Diagonal elements are not allowed");
        }

        Ai[array_index + 1] = Aj[array_index];
        Aj[array_index + 1] = Ai[array_index];

        if (fin.peek() != '\n')
            fin >> throwaway;
    }

    // automatically moves the vectors, no copying is done here
    return h_coo{n, 2*nnz, std::move(Ai), std::move(Aj)};
}

h_csr coo_to_csr(const h_coo &coo)
{

    h_csr sparse;
    sparse.rows = coo.n;
    sparse.cols = coo.n;

    sparse.nnz = coo.nnz;
    sparse.offsets.resize(coo.n + 1);
    sparse.positions.resize(coo.nnz);
    sparse.values.resize(coo.nnz);

    thrust::fill(sparse.offsets.begin(), sparse.offsets.end(), 0);
    thrust::fill(sparse.values.begin(), sparse.values.end(), 1.0f);

    for (int n = 0; n < coo.nnz; n++)
    {
        sparse.offsets[coo.Ai[n]]++;
    }

    for (int i = 0, cumsum = 0; i < coo.n; i++)
    {
        int temp = sparse.offsets[i];
        sparse.offsets[i] = cumsum;
        cumsum += temp;
    }
    sparse.offsets[coo.n] = coo.nnz;

    for (int n = 0; n < coo.nnz; n++)
    {
        int row = coo.Ai[n];
        int dest = sparse.offsets[row];

        sparse.positions[dest] = coo.Aj[n];

        sparse.offsets[row]++;
    }

    for (int i = 0, last = 0; i <= coo.n; i++)
    {
        int temp = sparse.offsets[i];
        sparse.offsets[i] = last;
        last = temp;
    }

    return sparse;
}

h_csr loadSymmFileToCsr(const std::string& filename) {
    h_coo coo = loadSymmFileToCoo(filename);
    return coo_to_csr(coo);
}