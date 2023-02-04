template <class I, class T, class T2>
void csr_add_csr_canonical(const I n_row, const I n_col,
                           const I Ap[], const I Aj[], const T Ax[],
                           const I Bp[], const I Bj[], const T Bx[],
                           I Cp[], I Cj[], T2 Cx[])
{
    // Method that works for canonical CSR matrices

    Cp[0] = 0;
    I nnz = 0;

    for (I i = 0; i < n_row; i++)
    {
        I A_pos = Ap[i];
        I B_pos = Bp[i];
        I A_end = Ap[i + 1];
        I B_end = Bp[i + 1];

        // while not finished with either row
        while (A_pos < A_end && B_pos < B_end)
        {
            I A_j = Aj[A_pos];
            I B_j = Bj[B_pos];

            if (A_j == B_j)
            {
                Cj[nnz] = A_j;
                Cx[nnz] = Ax[A_pos] + Bx[B_pos];
                A_pos++;
                B_pos++;
            }
            else if (A_j < B_j)
            {
                Cj[nnz] = A_j;
                Cx[nnz] = Ax[A_pos];
                A_pos++;
            }
            else
            {
                // B_j < A_j
                Cj[nnz] = B_j;
                Cx[nnz] = Bx[B_pos];
                B_pos++;
            }
            nnz++;
        }

        // tail
        while (A_pos < A_end)
        {
            Cj[nnz] = Aj[A_pos];
            Cx[nnz] = Ax[A_pos];
            nnz++;
            A_pos++;
        }
        while (B_pos < B_end)
        {
            Cj[nnz] = Bj[B_pos];
            Cx[nnz] = Bx[B_pos];
            nnz++;
            B_pos++;
        }

        Cp[i + 1] = nnz;
    }
}

template <class I, class T, class T2>
I csr_hadmul_csr_canonical(const I n_row, const I n_col,
                           const I Ap[], const I Aj[], const T Ax[],
                           const I Bp[], const I Bj[], const T Bx[],
                           I Cp[], I Cj[], T2 Cx[])
{
    // Method that works for canonical CSR matrices

    Cp[0] = 0;
    I nnz = 0;

    for (I i = 0; i < n_row; i++)
    {
        I A_pos = Ap[i];
        I B_pos = Bp[i];
        I A_end = Ap[i + 1];
        I B_end = Bp[i + 1];

        // while not finished with either row
        while (A_pos < A_end && B_pos < B_end)
        {
            I A_j = Aj[A_pos];
            I B_j = Bj[B_pos];

            if (A_j == B_j)
            {
                Cj[nnz] = A_j;
                Cx[nnz] = Ax[A_pos] * Bx[B_pos];
                nnz++;
                A_pos++;
                B_pos++;
            }
            else if (A_j < B_j)
            {
                A_pos++;
            }
            else
            {
                // B_j < A_j
                B_pos++;
            }
        }

        Cp[i + 1] = nnz;
    }

    return nnz;
}
