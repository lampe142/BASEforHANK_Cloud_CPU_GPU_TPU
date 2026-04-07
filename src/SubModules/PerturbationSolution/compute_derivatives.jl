using JLD2

"""
    compute_derivatives(sr, m_par, A, B; estim=false, only_SO=true, H_reduc=nothing, TO_reduc=nothing, fix_known_zeros=false)

Compute the derivatives of the equilibrium conditions `F` necessary for higher-order perturbation solutions.

This function performs the following steps:

 1. Calculates the first-order Jacobian of the (reduced) system and solves for the linear policy functions `gx` and `hx`.
 2. Computes the Hessian (second-order derivatives) of the system, potentially using a reduced-space approach for efficiency.
 3. Optionally computes the third-order derivatives (tensor).

It accounts for model reduction by differentiating the system with respect to the reduced factors and then mapping back to the full space.

# Arguments

  - `sr`: Steady-state results structure (`SteadyResults`).
  - `m_par`: Model parameters (`ModelParameters`).
  - `A`, `B`: Initial matrices for the Jacobians (often placeholders).
  - `estim`: Boolean (default `false`). Currently unused in the function body.
  - `only_SO`: Boolean (default `true`). If `true`, only derivatives up to the second order (Hessian) are computed. If `false`, third-order derivatives are also computed.
  - `H_reduc`: (Optional) Precomputed reduced Hessian matrix. If provided, skips calculation.
  - `TO_reduc`: (Optional) Precomputed reduced third-order derivative tensor.
  - `fix_known_zeros`: Boolean (default `false`). If `true`, utilizes knowledge of zero derivatives for certain distribution variables to speed up computation.
  - `buildpath`: String (default `""`). Path to save intermediate results like Hessians and third-order tensors.

# Returns

  - `gx`: Observation matrix (states to controls).
  - `hx`: State transition matrix.
  - `alarm_LinearSolution`: Boolean indicating if the linear solution failed.
  - `nk`: Number of stable eigenvalues.
  - `A`, `B`: First-order Jacobian matrices.
  - `H`: Second-order derivative matrix (Hessian).
  - `TO`: Third-order derivative matrix (only returned if `only_SO=false`).
"""
function compute_derivatives(
    sr::SteadyResults,
    m_par::ModelParameters,
    A::Array,
    B::Array;
    estim = false,
    only_SO = true,
    H_reduc = nothing,
    TO_reduc = nothing,
    fix_known_zeros = false,
    buildpath = "",
)
    transform_elements =
        transformation_elements(sr, sr.n_par.model, sr.n_par.distribution_states)
    ############################################################################
    # Check whether Steady state solves the difference equation
    ############################################################################
    length_X0 = sr.n_par.ntotal
    X0 = zeros(length_X0) .+ ForwardDiff.Dual(0.0, 0.0)

    F = Fsys(
        X0,
        X0,
        sr.XSS,
        m_par,
        sr.n_par,
        sr.indexes,
        sr.compressionIndexes,
        transform_elements,
        true,
    )

    FR = realpart.(F)

    idx_no_solution = findall((abs.(FR) .> 1e-6) .| (isnan.(FR) .| isinf.(FR)))
    if isempty(idx_no_solution)
        if sr.n_par.verbose
            @printf "Steady state is a solution!\n"
        end
    else
        @warn "Steady state is not a solution!"
        for i in idx_no_solution
            myfield = find_field_with_value(sr.indexes, i, false)
            @printf "Variable: %s, of index %i Value: %f\n" myfield i FR[i]
        end
    end

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################

    function Fsys_reduc(
        X::AbstractArray,
        XPrime::AbstractArray,
        XSS::Array{Float64,1},
        m_par::ModelParameters,
        n_par::NumericalParameters,
        indexes::IndexStruct,
        compressionIndexes::Vector,
        transform_elements::TransformationElements;
        ix_c = [],
        ix_cP = [],
    )

        # Leave variables with constant derivatives as zeros.
        ix_all = [i for i = 1:(n_par.ntotal_r)]
        X_old = zeros(eltype(X), n_par.ntotal_r)
        X_old[setdiff(ix_all, ix_c)] = X
        XPr_old = zeros(eltype(XPrime), n_par.ntotal_r)
        XPr_old[setdiff(ix_all, ix_cP)] = XPrime
        Y = n_par.PRightAll * X_old
        YPrime = n_par.PRightAll * XPr_old
        F = Fsys(
            Y,
            YPrime,
            XSS,
            m_par,
            n_par,
            indexes,
            compressionIndexes,
            transform_elements,
            true,
        )
        return n_par.PRightAll' * F
    end
    length_X0 = sr.n_par.ntotal_r
    X0 = zeros(length_X0) .+ ForwardDiff.Dual(0.0, 0.0)
    f_red(x, xP; ix_c = [], ix_cP = []) = Fsys_reduc(
        x,
        xP,
        sr.XSS,
        m_par,
        sr.n_par,
        sr.indexes,
        sr.compressionIndexes,
        transform_elements;
        ix_c = ix_c,
        ix_cP = ix_cP,
    )
    # first order
    @info "F.O.:"
    fFO_red(x) = f_red(x[1:length_X0], x[(length_X0 + 1):end])
    println("Calculating Jacobian of the reduced system (FO)")
    # BA = ForwardDiff.jacobian(fFO_red,zeros(2*length_X0))
    BA = @time ForwardDiff.jacobian(fFO_red, zeros(2 * length_X0))

    B = BA[:, 1:length_X0]
    A = BA[:, (length_X0 + 1):end]

    println("Solve Equation system (FO)")
    # gx, hx, alarm_LinearSolution, nk = SolveDiffEq(A, B, sr.n_par, estim;reduc_input=true)
    gx, hx, alarm_LinearSolution, nk = @time SolveDiffEq(A, B, sr.n_par)

    if alarm_LinearSolution
        return gx,
        hx,
        alarm_LinearSolution,
        nk,
        A,
        B,
        spzeros(length_X0 + 1, 4 * (length_X0 + 1)^2)
    end

    println("First Order State Space Solution Done")
    read_mem_linux()

    """
    Calculate Hessian (with accounting for any known zeros, additional to perturbation parameter)
    """
    if fix_known_zeros
        dist_indexes = vcat(
            [getfield(sr.indexes_r.distr, d) for d in propertynames(sr.indexes_r.distr)]...,
        )
        V_indexes = vcat(
            [
                getfield(sr.indexes_r.valueFunction, v) for
                v in propertynames(sr.indexes_r.valueFunction)
            ]...,
        )
    else
        dist_indexes = []
        V_indexes = []
    end
    ncP = sr.n_par.ncontrols_r # Number of ControlsPrime
    nc = sr.n_par.ncontrols_r - length(V_indexes) # Number of Controls
    nsP = sr.n_par.nstates_r - length(dist_indexes) # Number of StatesPrime
    ns = sr.n_par.nstates_r # Number of States
    ix_contrP = 1:ncP
    ix_contr = (ncP + 1):(ncP + nc)
    ix_statP = (ncP + nc + 1):(ncP + nc + nsP)
    ix_stat = (ncP + nc + nsP + 1):(ncP + nc + nsP + ns)
    # map SGU-ordering of variables to Levintal-ordering of variables
    f_lev_red(x) = f_red(
        [x[ix_stat]; x[ix_contr]],
        [x[ix_statP]; x[ix_contrP]];
        ix_c = V_indexes,
        ix_cP = dist_indexes,
    )
    fSO_red(x) = vec(ForwardDiff.jacobian(f_lev_red, x))
    x0 = zeros(2 * length_X0 - length(V_indexes) - length(dist_indexes))
    # Note: Chunk{2} is a value from experience, but may well be suboptimal
    cfg = ForwardDiff.JacobianConfig(fSO_red, x0, ForwardDiff.Chunk{2}())
    timestart = now()
    @info "SO:"
    println("start calculating reduced Hessian")
    println("using nested automatic differentiation")
    if !isnothing(H_reduc)
        H_red = H_reduc
    else
        # H_red is saved as H_red.jld2 in the current directory. If it exists, it will be overwritten.
        H_red = sparse(reshape(ForwardDiff.jacobian(fSO_red, x0, cfg), length_X0, :))
        println("save Hessian (H_red.jld2)")
        jldsave(joinpath(buildpath, "H_red.jld2"), true; H_red)
    end
    read_mem_linux()
    timer_help(timestart)
    ix_all = collect(1:(2 * (length_X0 + 1)))
    if (length(dist_indexes) > 0) & (length(V_indexes) > 0)
        ix_const = [
            sr.n_par.ncontrols_r .+ V_indexes .- sr.n_par.nstates_r
            2 * sr.n_par.ncontrols_r .+ dist_indexes
            2 * (length_X0 + 1)
            2 * sr.n_par.ncontrols_r + sr.n_par.nstates_r + 1
        ]
    elseif length(V_indexes) > 0
        ix_const = [
            sr.n_par.ncontrols_r .+ V_indexes .- sr.n_par.nstates_r
            2 * (length_X0 + 1)
            2 * sr.n_par.ncontrols_r + sr.n_par.nstates_r + 1
        ]
    else
        ix_const = [2 * (length_X0 + 1); 2 * sr.n_par.ncontrols_r + sr.n_par.nstates_r + 1]
    end

    # Fill parts of H that are not 0 (complement of {ix_c,ix_cP,σ,σP}) with HF_red
    timestart = now()
    H = fill_hessian(ix_all, ix_const, length_X0, H_red)
    @info "Hessian filled"
    timer_help(timestart)
    read_mem_linux()
    if !only_SO
        """
        Calculate third-order derivatives
        """
        fTO_red(x) = vec(ForwardDiff.jacobian(fSO_red, x))
        cfg = ForwardDiff.JacobianConfig(fTO_red, x0, ForwardDiff.Chunk{2}())
        timestart = now()
        @info "TO:"
        println("start calculating third order derivatives")
        if !isnothing(TO_reduc)
            println("load existing")
            TO_red = TO_reduc
        else
            TO_red = sparse(reshape(ForwardDiff.jacobian(fTO_red, x0, cfg), length_X0, :))
            jldsave(joinpath(buildpath, "TO_red.jld2"), true; TO_red)
        end
        read_mem_linux()
        timer_help(timestart)
        timestart = now()
        TO = fill_thirdDeriv(ix_all, ix_const, length_X0, TO_red)
        @info "Third order derivatives filled"
        timer_help(timestart)
        read_mem_linux()

        return gx, hx, alarm_LinearSolution, nk, A, B, H, TO
    else
        return gx, hx, alarm_LinearSolution, nk, A, B, H
    end
end

"""
    fill_hessian(ix_all, ix_const, length_X0, H_red)

Reconstruct the full Hessian matrix from the reduced Hessian.

This function expands the reduced Hessian (`H_red`) computed on the subset of active variables
back into the full sparse matrix format of the system's Hessian. It maps the indices of the
reduced system back to the full system, placing zeros where variables were held constant
(as specified by `ix_const`).

# Arguments

  - `ix_all`: Vector of all variable indices in the reduced system.
  - `ix_const`: Vector of indices for variables that were treated as constant (zero derivative).
  - `length_X0`: Number of equations/variables in the reduced system (dimensionality).
  - `H_red`: The sparse reduced Hessian matrix computed via automatic differentiation.

# Returns

  - `H`: The full sparse Hessian matrix of dimension `(length_X0 + 1) x (2 * (length_X0 + 1))^2`.
"""
function fill_hessian(
    ix_all::Vector{Int64},
    ix_const::Vector{Int64},
    length_X0::Number,
    H_red::SparseMatrixCSC{Float64},
)
    ix_tofill = setdiff(ix_all, ix_const)

    numnonzero = nnz(H_red)
    Hrows = Vector{Int64}(undef, numnonzero)
    Hcols = Vector{Int64}(undef, numnonzero)
    Hvals = Vector{Float64}(undef, numnonzero)
    i = 1
    @inbounds @views begin
        for (k1, j1) in enumerate(ix_tofill)
            for (k2, j2) in enumerate(ix_tofill)
                (row_nz, vals_nz) = findnz(H_red[:, (k1 - 1) * length(ix_tofill) + k2])
                lnz = length(row_nz)
                Hrows[i:(i + lnz - 1)] .= row_nz
                Hcols[i:(i + lnz - 1)] .= fill((j1 - 1) * 2 * (length_X0 + 1) + j2, lnz)
                Hvals[i:(i + lnz - 1)] .= vals_nz
                i += lnz
            end
        end
    end
    H = sparse(
        Hrows[1:(i - 1)],
        Hcols[1:(i - 1)],
        Hvals[1:(i - 1)],
        length_X0 + 1,
        4 * (length_X0 + 1)^2,
    )
    return H
end

"""
    fill_thirdDeriv(ix_all, ix_const, length_X0, TO_red)

Reconstruct the full third-order derivative tensor from the reduced tensor.

This function expands the reduced third-order derivative tensor (`TO_red`) back into the full
sparse matrix format. It is similar to `fill_hessian` but handles the three-dimensional
indexing inherent to third-order derivatives.

# Arguments

  - `ix_all`: Vector of all variable indices in the reduced system.
  - `ix_const`: Vector of indices for variables that were treated as constant.
  - `length_X0`: Number of equations/variables in the reduced system.
  - `TO_red`: The sparse reduced third-order derivative tensor.

# Returns

  - `TO`: The full sparse third-order derivative tensor.
"""
function fill_thirdDeriv(
    ix_all::Vector{Int64},
    ix_const::Vector{Int64},
    length_X0::Number,
    TO_red::SparseMatrixCSC{Float64},
)
    ix_tofill = setdiff(ix_all, ix_const)
    numnonzero = nnz(TO_red)
    TOrows = Vector{Int64}(undef, numnonzero)
    TOcols = Vector{Int64}(undef, numnonzero)
    TOvals = Vector{Float64}(undef, numnonzero)
    i = 1
    @inbounds @views begin
        for (k1, j1) in enumerate(ix_tofill)
            for (k2, j2) in enumerate(ix_tofill)
                for (k3, j3) in enumerate(ix_tofill)
                    (row_nz, vals_nz) = findnz(
                        TO_red[
                            :,
                            (k1 - 1) * length(ix_tofill)^2 + (k2 - 1) * length(ix_tofill) + k3,
                        ],
                    )
                    lnz = length(row_nz)
                    TOrows[i:(i + lnz - 1)] .= row_nz
                    TOcols[i:(i + lnz - 1)] .= fill(
                        (j1 - 1) * (2 * (length_X0 + 1))^2 +
                        (j2 - 1) * (2 * (length_X0 + 1)) +
                        j3,
                        lnz,
                    )
                    TOvals[i:(i + lnz - 1)] .= vals_nz
                    i += lnz
                end
            end
        end
    end
    TO = sparse(
        TOrows[1:(i - 1)],
        TOcols[1:(i - 1)],
        TOvals[1:(i - 1)],
        length_X0 + 1,
        (2 * (length_X0 + 1))^3,
    )
    return TO
end
