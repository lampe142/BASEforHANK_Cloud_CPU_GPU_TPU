using LinearAlgebra, SparseArrays

"""
    stepwiseLSKron(M,A,B)

Calculate kron(A,B)*M stepwise, without explicit computation of kron(A,B). Can use that B,M is sparse.
"""
function stepwiseLSKron(M, A, B)
    (q, n) = size(A)
    (p, k) = size(B)
    (m, l) = size(M)
    aux_1 = zeros(p, q, l)
    Mres = reshape(M, k, n, l)
    aux_2 = zeros(p, q)
    Acol = collect(A)
    for i = 1:l
        BLAS.gemm!('N', 'T', 1.0, collect(B * Mres[:, :, i]), Acol, 0.0, aux_2)
        #m_aux = reshape(M[i,:],n,k)
        aux_1[:, :, i] .= aux_2
    end
    return reshape(aux_1, q * p, l)
end

"""
    stepwiseRSKron(M,A,B)

Calculate M*kron(A,B) stepwise, without explicit computation of kron(A,B). Can use that B,M is sparse.
"""
function stepwiseRSKron(M, A, B)
    return collect(stepwiseLSKron(M', A', B')')
end

"""
    stepwiseKronecker2(M,A,B)

Calculate M*kron(A,B) stepwise, without explicit computation of kron(A,B), using parallelization
"""
function stepwiseKronecker2(M, A, B)
    (m_M, n_M) = size(M)
    (m_A, n_A) = size(A)
    (m_B, n_B) = size(B)
    k = Int(n_M / m_B)
    if k != m_A
        error("dimensions for stepwise Kronecker do not fit")
    end
    aux_1 = Matrix{Float64}(undef, m_M * n_B, k)
    Threads.@threads for j = 1:k
        aux_1[:, j] = vec(M[:, (m_B * (j - 1) + 1):(m_B * j)] * B)
    end
    aux_1 = dropzeros(sparse(aux_1))
    aux_2 = Matrix{Float64}(undef, m_M, n_A * n_B)
    Threads.@threads for j = 1:n_A
        aux_2[:, ((j - 1) * n_B + 1):(j * n_B)] = reshape(aux_1 * A[:, j], m_M, n_B)
    end
    return dropzeros(sparse(aux_2))
end

"""
    stepwiseLSKronecker2(M,A,B)

Calculate kron(A,B)*M stepwise, without explicit computation of kron(A,B), using parallelization
"""
function stepwiseLSKronecker2(M, A, B)
    m = size(M, 1)
    n = size(A, 2)
    q = size(A, 1)
    p = size(B, 1)
    l = size(M, 2)
    k = Int(m / n)
    aux_1 = Matrix{Float64}(undef, n, p * l)
    Threads.@threads for j = 1:n
        aux_1[j, :] = vec(B * M[(k * (j - 1) + 1):(k * j), :])'
    end
    aux_1 = dropzeros(sparse(aux_1))
    aux_2 = Matrix{Float64}(undef, q * p, l)
    Threads.@threads for j = 1:q
        aux_2[((j - 1) * p + 1):(j * p), :] = reshape(A[j, :]' * aux_1, p, l)
    end
    return dropzeros(sparse(aux_2))
end

"""
    stepwiseLSBsparseKronecker2(M,A,B)

Calculate kron(A,B)*M stepwise, without explicit computation of kron(A,B), using parallelization.
Use that B has many zero rows.
"""
function stepwiseLSBsparseKronecker2(M, A, B)
    m = size(M, 1)
    n = size(A, 2)
    q = size(A, 1)
    p = size(B, 1)
    # find zero rows in B
    (Inz, ~, ~) = findnz(sparse(B))
    Inz = unique(Inz)
    pnz = length(Inz)
    l = size(M, 2)
    k = Int(m / n)
    aux_1_red = Matrix{Float64}(undef, n, pnz * l)
    Threads.@threads for j = 1:n
        aux_1_red[j, :] = vec(B[Inz, :] * M[(k * (j - 1) + 1):(k * j), :])'
    end
    rowIX = Int64[]
    colIX = Int64[]
    val = Float64[]
    for j = 1:q
        append!(rowIX, repeat((j - 1) * p .+ Inz; outer = l))
        append!(colIX, repeat(1:l; inner = pnz))
        append!(val, (A[j, :]' * aux_1_red)[:])
    end
    return dropzeros(sparse(rowIX, colIX, val, q * p, l))
end
