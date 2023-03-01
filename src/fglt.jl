using Pkg
Pkg.activate(".")
Pkg.instantiate()

Pkg.add("MatrixMarket")
Pkg.add("CUDA")
Pkg.add("BenchmarkTools")

using SparseArrays
using MatrixMarket
using CUDA
using CUDA.CUSPARSE

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

# read arguments
file_name = ARGS[1]
file_name = "./datasets/s6.mtx"
# check if file exists

if !isfile(file_name)
    println("File does not exist")
    exit(1)
end

A = mmread(file_name)

warm_up_cuda = CuArray{Int32}(undef, 10)

# save time for d_A in a variable
using BenchmarkTools

CuSparseMatrixCSR{Int32}(A) # precompile
HOST_TO_DEVICE = @elapsed (d_A = CuSparseMatrixCSR{Int32}(A))

fglt(d_A) # precompile
FGLT = @elapsed ((p1, d2, d3, c3) = fglt(d_A))

h_p1 = Array(p1) # precompile
DEVICE_TO_HOST = @elapsed (
    h_p1 = Array(p1),
    h_d2 = Array(d2),
    h_d3 = Array(d3),
    h_d4 = Array(c3)
)

println("$(HOST_TO_DEVICE*1e6)\t$(FGLT*1e6)\t$(DEVICE_TO_HOST*1e6)")
