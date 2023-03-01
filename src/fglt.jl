using SparseArrays
using MatrixMarket
using CUDA
using CUDA.CUSPARSE

# Load the matrix
A = mmread("../datasets/com-Youtube.mtx")
# A = mmread("s6.mtx")

d_A = CuSparseMatrixCSR{Int32}(A)

n = size(A, 1)

# CUSPARSE SPMV:

# Create the handle

function c3_kernel!(offsets, positions, c3)

    n = length(offsets) - 1

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i = index:stride:n
        @inbounds i_nb_start = offsets[i]
        @inbounds i_nb_end = offsets[i+1]

        amt_i = 0
        for i_nb_idx = i_nb_start:(i_nb_end-1)
            @inbounds j = positions[i_nb_idx]

            if i < j
                break
            end

            @inbounds j_nb_start = offsets[j]
            @inbounds j_nb_end = offsets[j+1]

            _i_nb_idx = i_nb_start
            _j_nb_idx = j_nb_start

            amt_j = 0

            while _i_nb_idx < i_nb_end && _j_nb_idx < j_nb_end
                @inbounds _i_nb_pos = positions[_i_nb_idx]
                @inbounds _j_nb_pos = positions[_j_nb_idx]

                if _i_nb_pos >= i || _j_nb_pos >= j
                break
                end

                if _i_nb_pos > _j_nb_pos
                    _j_nb_idx += 1
                elseif _i_nb_pos < _j_nb_pos
                    _i_nb_idx += 1
                else
                    @inbounds CUDA.@atomic c3[_i_nb_pos] += 1
                    amt_j += 1
                    _i_nb_idx += 1
                    _j_nb_idx += 1
                end
            end

            amt_i += amt_j

            @inbounds CUDA.@atomic c3[j] += amt_j
        end

        @inbounds CUDA.@atomic c3[i] += amt_i
    end
    return nothing
end

# y = Ax
function spmv_symbolic_kernel!(offsets, positions, x, y)
    n = length(x)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i = index:stride:n
        sum::Int32 = 0
        for j = offsets[i]:(offsets[i+1]-1)
            sum += x[positions[j]]
        end
        y[i] = sum
    end
    return nothing
end

function fglt(A::CuSparseMatrixCSR{Int32})
    n = size(A, 1)

    p1 = A.rowPtr[2:end] - A.rowPtr[1:end-1]

    t = 512
    b = cld(n, t)

    c3 = CUDA.zeros(Int32, d_A.dims[1])
    CUDA.@sync @cuda(
        threads=t,
        blocks=b,
        c3_kernel!(A.rowPtr, A.colVal, c3)
    )

    Axp1 = CuArray{Int32}(undef, n)
    CUDA.@sync @cuda(
        threads=t,
        blocks=b,
        spmv_symbolic_kernel!(A.rowPtr, A.colVal, p1, Axp1)
    )

    p2 = Axp1 - p1

    d2 = p2 - 2 * c3

    d3 = (p1) .* (p1 .- 1) .- c3

    return p1, d2, d3, c3
end

p1, d2, d3, c3 = fglt(d_A)
