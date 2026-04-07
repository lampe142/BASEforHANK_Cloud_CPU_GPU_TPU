"""
    doublingGxx(iAB, hx, iAD; tol=1e-14, maxit=20)

Solve a Sylvester-like equation using the doubling algorithm (Kim–Kim–Sims, JEDC 2008).
Computes `X` satisfying `X = reshape(M*hx, size(iAD)) + X_old` with powers of `hx` and `iAB`.
Returns the solution matrix `X`. Prints convergence diagnostics; warns if not converged.
"""
function doublingGxx(iAB, hx, iAD, tol::Float64 = 1.e-14, maxit::Int64 = 20)
    m = size(iAB, 1)
    n = size(hx, 1)
    X = -iAD
    kk = 0
    cc = 1 + tol
    M = Matrix{Float64}(undef, m * n, n)
    X_old = copy(X)
    iABpower = -iAB
    hxpower = hx
    while (kk <= maxit && cc > tol)
        X_old[:, :] = X[:, :]
        # Compute iABpower*X times kron(hxpower,hxpower) w/o computing kron() first
        iABX = iABpower * X
        Threads.@threads for j = 1:n
            M[:, j] = vec(iABX[:, (n * (j - 1) + 1):(n * j)] * hxpower)
        end
        X = reshape(M * hxpower, size(iAD)) + X_old
        hxpower = hxpower^2
        iABpower = iABpower^2
        cc = maximum(sum(abs.(X - X_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "doublingGxx:: Convergence not achieved of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return X
end

"""
    doublingExfkr2(hx, c; tol=eps(), maxit=20)

Second-order doubling scheme using stepwise Kronecker products to avoid explicit large kron.
Updates `X` from seed `c` with repeated powers of `hx`. Returns `X`; prints timing and divergence.
"""
function doublingExfkr2(hx, c, tol::Float64 = eps(), maxit::Int64 = 20)
    n = size(hx, 1)
    X = c
    kk = 0
    cc = 1 + tol
    M = Matrix{Float64}(undef, n^3, n)
    X_old = copy(X)
    hxpower = hx
    timer = now()
    while (kk <= maxit && cc > tol)
        X_old[:, :] = X[:, :]
        # Compute kron(hxpower,hxpower)*X times kron(hxpower',hxpower') w/o computing kron() first
        krhxphxpX = stepwiseLSKron(X, hxpower, hxpower)
        Threads.@threads for j = 1:n
            M[:, j] = vec(krhxphxpX[:, (n * (j - 1) + 1):(n * j)] * hxpower')
        end
        X = reshape(M * hxpower', size(c)) + X_old
        hxpower = hxpower^2
        cc = maximum(sum(abs.(X - X_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
        println("Time elapsed: ", now() - timer)
        timer = now()
    end
    if cc > tol
        println(
            string(
                "doublingGxx:: Convergence not achieved of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return X
end

"""
    doublingGxxx(iAB, hx, iAD; tol=eps(), maxit=20)

Third-order variant of the doubling algorithm for Sylvester-type equations with triple kron structure.
Returns the solution matrix `X`; prints convergence diagnostics.
"""
function doublingGxxx(iAB, hx, iAD, tol::Float64 = eps(), maxit::Int64 = 20)
    m = size(iAB, 1)
    n = size(hx, 1)
    X = -iAD
    kk = 0
    cc = 1 + tol
    P = Matrix{Float64}(undef, m * n, n)
    M = Matrix{Float64}(undef, m * (n^2), n)
    X_old = copy(X)
    iABpower = -iAB
    hxpower = hx
    while (kk <= maxit && cc > tol)
        X_old[:, :] = X[:, :]
        # Compute iABpower*X times kron(hxpower,hxpower,hxpower) w/o computing kron() first
        iABX = iABpower * X
        for k = 1:n
            Threads.@threads for j = 1:n
                #for j=1:n
                P[:, j] = vec(
                    iABX[:, (n^2 * (k - 1) + n * (j - 1) + 1):(n^2 * (k - 1) + n * j)] *
                    hxpower,
                )
            end
            M[:, k] = vec(P * hxpower)
        end
        X = reshape(M * hxpower, size(iAD)) + X_old
        hxpower = hxpower^2
        iABpower = iABpower^2
        cc = maximum(sum(abs.(X - X_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "doublingGxx:: Convergence not achieved of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return X
end

"""
    doublingSimple(A, Fkron, B; tol=eps(), maxit=20)

Simple doubling iteration `X = X_old + A^p * X_old * Fkron^p` to solve `A*X*Fkron + B = 0`.
Returns `X`; prints iteration divergence and warns on non-convergence.
"""
function doublingSimple(A, Fkron, B, tol::Float64 = eps(), maxit::Int64 = 20)
    X = -B
    kk = 0
    cc = 1 + tol
    X_old = copy(X)
    Apower = A
    Fkronpower = Fkron
    while (kk <= maxit && cc > tol)
        X_old[:, :] = X[:, :]
        X = X_old + Apower * X_old * Fkronpower
        Fkronpower = Fkronpower^2
        Apower = Apower^2
        cc = maximum(sum(abs.(X - X_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "doublingSimple:: Convergence not achieved of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return X
end

"""
    doublingSO(hx, c; tol=eps(), maxit=20)

Sparse second-order doubling using structured left-stepwise Kronecker operations.
Seeds from `c` and returns sparse `X`; prints divergence and warnings when not converged.
"""
function doublingSO(hx, c, tol::Float64 = eps(), maxit::Int64 = 20)
    n = size(hx, 1)
    X = sparse(c)
    kk = 0
    cc = 1 + tol
    X_old = copy(X)
    hxpower = hx
    while (kk <= maxit && cc > tol)
        X_old[:] = X[:]
        alpha = stepwiseLSKron(X, hxpower, hxpower)
        X = sparse(X_old + alpha)
        hxpower = hxpower^2
        cc = maximum(sum(abs.(X - X_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "doublingTO:: Convergence not achieved of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return X
end

"""
    doublingTO(hx, c; tol=eps(), maxit=20)

Sparse third-order (two kron then one) doubling using structured Kronecker operations.
Seeds from `c` and returns sparse `X`; prints divergence and warnings when not converged.
"""
function doublingTO(hx, c, tol::Float64 = eps(), maxit::Int64 = 20)
    n = size(hx, 1)
    X = sparse(c)
    kk = 0
    cc = 1 + tol
    X_old = copy(X)
    hxpower = hx
    while (kk <= maxit && cc > tol)
        X_old[:] = X[:]
        alpha = stepwiseLSKron(X, kron(hxpower, hxpower), hxpower)
        X = sparse(X_old + alpha)
        hxpower = hxpower^2
        cc = maximum(sum(abs.(X - X_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "doublingTO:: Convergence not achieved of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return X
end

"""
    itGxx(iABred, hx, iADred, W2, lowerX=nothing; tol=1e-6, maxit=1000)

Fixed-point iteration for a reduced Sylvester equation using blockwise updates without explicit kron.
If `lowerX` is not provided, starts from zeros of `size(iADred)`. Returns the updated `lowerX`.
"""
function itGxx(
    iABred,
    hx,
    iADred,
    W2,
    lowerX = nothing,
    tol::Float64 = 1.e-6,
    maxit::Int64 = 1000,
)
    m = size(iABred, 1)
    n = size(hx, 1)
    if nothing == lowerX
        lowerX = zeros(size(iADred))
    end
    println("itGxx: Size of matrix that is iterated over: ", size(lowerX))
    kk = 0
    cc = 1 + tol
    M = Matrix{Float64}(undef, m * n, n)
    # Size of successively smaller blocks
    bix(j) = Int(j * (n - (j - 1) / 2))
    lowerX_old = copy(lowerX)
    @views @inbounds while (kk <= maxit && cc > tol)
        lowerX_old[:, :] = lowerX[:, :]
        iABX = iABred * lowerX * W2
        Threads.@threads for j = 1:n
            M[:, j] = vec(iABX[:, (n * (j - 1) + 1):(n * j)] * hx)
        end
        Threads.@threads for j = 1:n
            cix = (bix(j - 1) + 1):bix(j)
            lowerX[:, cix] = -(
                reshape(M[(m * (j - 1) + 1):end, :] * hx[:, j], m, n - j + 1) +
                iADred[:, cix]
            )
        end
        cc = maximum(sum(abs.(lowerX - lowerX_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "fastgensylv:: Convergence not achieved in fixed point solution of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return lowerX
end

"""
    itLowerX(iABred, C, iADred, lowerX=nothing; tol=1e-6, maxit=1000)

Simple fixed-point iteration to solve `iABred*X*C + iADred + X = 0`.
Starts from zeros if `lowerX` is `nothing`. Returns the iterated `lowerX`.
"""
function itLowerX(
    iABred,
    C,
    iADred,
    lowerX = nothing,
    tol::Float64 = 1.e-6,
    maxit::Int64 = 1000,
)
    m = size(iABred, 1)
    n = size(C, 1)

    if nothing == lowerX
        lowerX = zeros(m, n)
    end
    println("itLowerX: Size of matrix that is iterated over: ", size(lowerX))

    kk = 0
    cc = 1 + tol
    @inbounds while (kk <= maxit && cc > tol)
        lowerX_old = lowerX
        lowerX = -(iABred * lowerX * C + iADred)
        cc = maximum(sum(abs.(lowerX - lowerX_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "fastgensylv:: Convergence not achieved in fixed point solution of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return lowerX
end

"""
    itLowerXkron3(iABred, hx, iADred, lowerX=nothing; tol=1e-6, maxit=1000)

Fixed-point iteration for a third-order kron-structured Sylvester system.
Updates `lowerX` using nested block operations with `hx`. Returns the iterated `lowerX`.
"""
function itLowerXkron3(
    iABred,
    hx,
    iADred,
    lowerX = nothing,
    tol::Float64 = 1.e-6,
    maxit::Int64 = 1000,
)
    m = size(iABred, 1)
    n = size(hx, 1)

    if isnothing(lowerX)
        lowerX = zeros(m, n^3)
    end
    P = Matrix{Float64}(undef, m * n, n)
    M = Matrix{Float64}(undef, m * (n^2), n)
    println("itLowerX: Size of matrix that is iterated over: ", size(lowerX))
    lowerX_old = copy(lowerX)
    kk = 0
    cc = 1 + tol
    @inbounds while (kk <= maxit && cc > tol)
        lowerX_old[:, :] = lowerX[:, :]
        iABX = iABred * lowerX
        for k = 1:n
            Threads.@threads for j = 1:n
                P[:, j] = vec(
                    iABX[:, (n^2 * (k - 1) + n * (j - 1) + 1):(n^2 * (k - 1) + n * j)] * hx,
                )
            end
            M[:, k] = vec(P * hx)
        end
        lowerX = -(reshape(M * hx, size(iADred)) + iADred)
        cc = maximum(sum(abs.(lowerX - lowerX_old); dims = 1))
        kk = kk + 1
        println("it. ", kk, ": divergence ", cc)
    end
    if cc > tol
        println(
            string(
                "fastgensylv:: Convergence not achieved in fixed point solution of Sylvester equation after ",
                maxit,
                " iterations",
            ),
        )
    end
    return lowerX
end
