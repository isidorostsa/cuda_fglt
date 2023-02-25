#include "host_structs.hpp"
#include "printing.hpp"

/* Cool but not used
h_csr::operator d_cusparse_csr() const{
    return d_cusparse_csr(rows, cols, nnz, offsets, positions, values);
}
*/

h_csr::h_csr() : h_csr(0, 0, 0) {}

h_csr::h_csr(int rows, int cols, int nnz)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets = thrust::host_vector<int>(rows + 1);
    this->positions = thrust::host_vector<int>(nnz);
    this->values = thrust::host_vector<float>(nnz);
}

h_csr::h_csr(int rows, int cols, int nnz, thrust::host_vector<int> &offsets, thrust::host_vector<int> &positions, thrust::host_vector<float> &values)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets = offsets;
    this->positions = positions;
    this->values = values;
}

h_csr::h_csr(int rows, int cols, int nnz, thrust::host_vector<int> &&offsets, thrust::host_vector<int> &&positions, thrust::host_vector<float> &&values)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets = std::move(offsets);
    this->positions = std::move(positions);
    this->values = std::move(values);
}

// move constructor
h_csr::h_csr(h_csr &&other) : h_csr(other.rows, other.cols, other.nnz, std::move(other.offsets), std::move(other.positions), std::move(other.values)) {}

// conversion constructor
h_csr::h_csr(const d_cusparse_csr &d_csr) : rows(d_csr.get_rows()), cols(d_csr.get_cols()), nnz(d_csr.get_nnz()),
                                            offsets(d_csr.get_offsets()), positions(d_csr.get_positions()), values(d_csr.get_values()) {}

// getters
int h_csr::get_rows() const
{
    return rows;
}

int h_csr::get_cols() const
{
    return cols;
}

int h_csr::get_nnz() const
{
    return nnz;
}

const thrust::host_vector<int>& h_csr::get_offsets() const
{
    return offsets;
}

const thrust::host_vector<int>& h_csr::get_positions() const
{
    return positions;
}

const thrust::host_vector<float>& h_csr::get_values() const
{
    return values;
}

void h_csr::resize(int rows, int cols, int nnz)
{
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    this->offsets.resize(rows + 1);
    this->positions.resize(nnz);
    this->values.resize(nnz);
}

h_csr h_csr::hadamard(const h_csr &a, const h_csr &b)
{
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    int rows = a.rows;
    int cols = a.cols;
    int nnz = 0;

    // max size due to multiplication
    h_csr c(rows, cols, thrust::max(a.nnz, b.nnz));

    c.offsets[0] = 0;

    for (int i = 0; i < rows; i++)
    {
        int a_pos = a.offsets[i];
        int b_pos = b.offsets[i];

        int b_end = b.offsets[i + 1];
        int a_end = a.offsets[i + 1];

        while (a_pos < a_end && b_pos < b_end)
        {
            if (a.positions[a_pos] == b.positions[b_pos])
            {
                c.positions[nnz] = a.positions[a_pos];
                c.values[nnz] = a.values[a_pos] * b.values[b_pos];
                nnz++;
                a_pos++;
                b_pos++;
            }
            else if (a.positions[a_pos] < b.positions[b_pos])
            {
                a_pos++;
            }
            else
            {
                b_pos++;
            }
        }

        c.offsets[i + 1] = nnz;
    }

    c.resize(rows, cols, nnz);

    return c;
}