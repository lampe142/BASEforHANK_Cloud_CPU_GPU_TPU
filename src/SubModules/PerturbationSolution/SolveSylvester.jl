using Dates

"""
    SolveSylvester(gx, hx, B, A, H, n_par, indexes, covariances, shock_names; Xred=nothing, TestOfGoodness=false)

Solve a generalized Sylvester equation to compute second-order approximations of policy
functions and risk adjustments.

This function computes the second-order derivatives of the policy functions (`Gxx`, `Hxx`)
and the risk adjustment terms (`Gσσ`, `Hσσ`) for the observation and state transition
equations. The notation and method follow Levintal (2017). The system to solve is of the
form `A*X + B*X*C + D = 0`.

# Arguments

  - `gx`: First-order derivatives of the observation equation (control variables).
  - `hx`: First-order derivatives of the state transition equation (state variables).
  - `B`, `A`: Jacobians of the equilibrium conditions (`B` w.r.t `X`, `A` w.r.t `XPrime`).
  - `H`: Hessian of the equilibrium conditions (second-order derivatives).
  - `n_par`: Numerical parameters structure.
  - `indexes`: Index structure for variables.
  - `covariances`: Covariance matrix of the shocks.
  - `shock_names`: Vector of shock names.
  - `Xred`: (Optional) Initial guess for the solution matrix `X`. If `nothing`, it is
    initialized automatically.
  - `TestOfGoodness`: (Optional) If `true`, computes and prints residuals to test the
    accuracy of the solution.

# Returns

  - `Gxx`: Second-order derivatives of the observation equation (controls).
  - `Hxx`: Second-order derivatives of the state transition equation (states).
  - `Gσσ`: Second-order derivatives with respect to the volatility parameter σ (risk
    adjustment for controls).
  - `Hσσ`: Second-order derivatives with respect to the volatility parameter σ (risk
    adjustment for states).
  - `Xred`: The full solution matrix `X` (useful for warm-starting subsequent calls).
"""
function SolveSylvester(
    gx,
    hx,
    B,
    A,
    H,
    n_par,
    indexes,
    covariances,
    shock_names,
    Xred = nothing;
    TestOfGoodness = false,
)
    Fy = B[:, (n_par.nstates + 1):end]
    FxP = A[:, 1:(n_par.nstates)]
    FyP = A[:, (n_par.nstates + 1):end]
    nshocks = size(covariances, 1)
    # Compute A2
    timer1 = now()
    vx0 = [
        [sparse(gx * hx) spzeros(n_par.ncontrols)]
        [sparse(gx) spzeros(n_par.ncontrols)]
        [[sparse(hx) spzeros(n_par.nstates)]; [spzeros(1, n_par.nstates) 1]]
        sparse(1.0I, n_par.nstates + 1, n_par.nstates + 1)
    ]
    A = Matrix{Float64}(undef, n_par.ntotal * (n_par.nstates + 1), 2 * (n_par.ntotal + 1))
    Threads.@threads for j = 1:(2 * (n_par.ntotal + 1))
        A[:, j] = vec(
            H[
                1:(end - 1),
                (2 * (n_par.ntotal + 1) * (j - 1) + 1):(2 * (n_par.ntotal + 1) * j),
            ] * vx0,
        )
    end
    A = sparse(A)
    # B2 contains the sum of the Kronecker-square of the first-order derivative of h and of shock variances.
    B = Matrix{Float64}(undef, n_par.ntotal, (n_par.nstates + 1)^2)
    read_mem_linux()
    timer_help(timer1)
    ##########
    # Different way of calculating (slower): B = reshape(A*vx0,nx+ny,(nx+1)^2)
    timer2 = now()
    Threads.@threads for j = 1:(n_par.nstates + 1)
        B[:, ((j - 1) * (n_par.nstates + 1) + 1):(j * (n_par.nstates + 1))] =
            reshape(A * vx0[:, j], n_par.ntotal, n_par.nstates + 1)
    end
    B = sparse(B)
    vx1 = [
        [sparse(gx) spzeros(n_par.ncontrols)]
        spzeros(n_par.ncontrols, n_par.nstates + 1)
        sparse(1.0I, n_par.nstates + 1, n_par.nstates + 1)
        spzeros(n_par.nstates + 1, n_par.nstates + 1)
    ]
    A = Matrix{Float64}(undef, (n_par.ntotal) * (n_par.nstates + 1), 2 * (n_par.ntotal + 1))
    Threads.@threads for j = 1:(2 * (n_par.ntotal + 1))
        A[:, j] = vec(
            H[
                1:(end - 1),
                (2 * (n_par.ntotal + 1) * (j - 1) + 1):(2 * (n_par.ntotal + 1) * j),
            ] * vx1,
        )
    end
    A = sparse(A)
    C = Matrix{Float64}(undef, n_par.ntotal, (n_par.nstates + 1)^2)
    read_mem_linux()
    timer_help(timer2)
    ##########
    # Different way of calculating (slower): C = reshape(A*vx1,nx+ny,(nx+1)^2)
    timer3 = now()
    Threads.@threads for j = 1:(n_par.nstates + 1)
        C[:, ((j - 1) * (n_par.nstates + 1) + 1):(j * (n_par.nstates + 1))] =
            reshape(A * vx1[:, j], n_par.ntotal, n_par.nstates + 1)
    end
    C = sparse(C)
    η = sparse(
        [getfield(indexes, s) for s in shock_names],
        collect(1:nshocks),
        ones(nshocks),
        n_par.nstates + 1,
        nshocks,
    )
    Eζ2 = [spzeros((n_par.nstates + 1)^2, (n_par.nstates + 1)^2 - 1) kron(η, η) * reshape(
        covariances,
        (:, 1),
    )]
    read_mem_linux()
    timer_help(timer3)
    A2 = B + C * Eζ2
    println("computed A2")
    Z21 = kron(
        sparse(1.0I, n_par.nstates + 1, n_par.nstates),
        sparse(1.0I, n_par.nstates + 1, n_par.nstates),
    )
    A2Z21c = A2 * Z21

    A = [FxP + FyP * gx Fy]
    B = [zeros(n_par.ntotal, n_par.nstates) FyP]

    ## To solve: Generalized Sylvester equation
    ## Set X = [Hxx; Gxx]
    ## Then X has to solve A*X + B*X*C2 + A2Z21 = 0

    # Set maximum threads (for speedy matrix multiplication)
    BLAS.set_num_threads(Threads.nthreads())

    iAB = A \ B
    println("computed iAB")
    read_mem_linux()
    # Identify 0-columns of iAB
    iABzero = Int[]
    for j = 1:(n_par.ntotal)
        if 0 == sum(abs.(iAB[:, j]))
            append!(iABzero, j)
        end
    end
    iABix = setdiff(1:(n_par.ntotal), iABzero)
    X = zeros(size(A2Z21c))
    Dc = collect(A2Z21c)
    iADc = A \ Dc
    println("computed iADc")
    read_mem_linux()

    if nothing == Xred
        Xred = X[iABix, :]
    end
    # stop time!
    timer = time()
    Xred = doublingGxx(iAB[iABix, iABix], hx, iADc[iABix, :])
    time_solveapr = time() - timer
    println("time used for iteration: ", time_solveapr)
    read_mem_linux()

    # retrieve rest of X
    timer4 = now()
    read_mem_linux()
    X[iABix, :] = Xred
    X[iABzero, :] = -(stepwiseRSKron(iAB[iABzero, iABix] * Xred, hx, hx) + iADc[iABzero, :])
    timer_help(timer4)

    if TestOfGoodness
        println("test goodness: ")
        # Test goodness of approximation
        res = A * X + stepwiseRSKron(B * X, hx, hx) + A2 * Z21
        println(
            "max. colsum of res: ",
            maximum(sum(abs.(res); dims = 1)),
            ", max. abs. entry of res: ",
            norm(res, Inf),
        )
    end
    println("risk adjustment")
    read_mem_linux()
    Hxx = X[1:(n_par.nstates), :]
    Gxx = X[(n_par.nstates + 1):end, :]
    # Solve linear equation system for Gσσ, Hσσ
    Eηϵ2 = kron(η[1:(end - 1), :], η[1:(end - 1), :]) * reshape(covariances, (:, 1))
    c = collect(A2[:, end] + FyP * Gxx * Eηϵ2)
    D = collect([FxP + FyP * gx FyP + Fy])
    Xσ = -D \ c
    Hσσ = Xσ[1:(n_par.nstates)]
    Gσσ = Xσ[(n_par.nstates + 1):end]
    println("risk adjustment done")
    read_mem_linux()
    return Gxx, Hxx, Gσσ, Hσσ, Xred
end
