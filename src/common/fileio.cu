#include <iostream>
#include <string>
#include <fstream>

#include <thrust/host_vector.h>
#include "fileio.hpp"

#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wunused-result"

Coo_matrix loadFileToCoo(const std::string filename)
{
    std::ifstream fin(filename);

    int n, nnz;
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');

    fin >> n >> n >> nnz;

    thrust::host_vector<int> Ai(nnz);
    thrust::host_vector<int> Aj(nnz);

    int throwaway;
    // lines may be of the form: i j or i j throwaway
    for (int i = 0; i < nnz; ++i)
    {
        fin >> Ai[i] >> Aj[i];
        Ai[i]--;
        Aj[i]--;
        if (fin.peek() != '\n')
            fin >> throwaway;
    }

    // automatically moves the vectors, no copying is done here
    return Coo_matrix{n, nnz, std::move(Ai), std::move(Aj)};
}

h_csr loadFileToCsr(const std::string filename)
{
    // check file exists
    if (!std::ifstream(filename).good())
    {
        std::cout << "File " << filename << " does not exist" << std::endl;
        exit(1);
    }

    FILE *fin = fopen(filename.c_str(), "r");

    std::cout << "Opened " << filename << std::endl;

    while (fgetc(fin) == '%')
    {
        while (fgetc(fin) != '\n')
        {
            // do nothing
        };
    } // put last character back
    fseek(fin, -1, SEEK_CUR);

    int n, nnz;
    fscanf(fin, "%d %d %d", &n, &n, &nnz);
    thrust::host_vector<int> offsets(n + 1, 0);
    thrust::host_vector<int> positions(nnz);

    int i, j;
    // lines may be of the form: i j or i j throwaway where throwaway can be any number of characters until a newline
    for (int ind = 0; ind < nnz; ++ind)
    {
        fscanf(fin, "%d %d", &i, &j);
        --i;
        --j;

        positions[ind] = i;
        offsets[j + 1]++;

        // skip the rest of the line
        // unless we are at the end of the file
        if (ind < nnz - 1)
        {
            while (fgetc(fin) != '\n')
            {
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        offsets[i + 1] += offsets[i];
    }

    // automatically moves the vectors, no copying is done here
    return h_csr(n, n, nnz, std::move(offsets), std::move(positions), thrust::host_vector<float>(nnz, 1.0f));
}

void coo_to_sparse(const Coo_matrix &coo, h_csr &sparse)
{
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
}