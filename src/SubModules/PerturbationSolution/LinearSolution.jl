"""
    LinearSolution(sr, m_par, A, B; allow_approx_sol=true, ss_only=false)

Calculate the linearized solution to the non-linear difference equations defined by function
[`Fsys()`](@ref), using Schmitt-Grohé & Uribe (JEDC 2004) style linearization (apply the
implicit function theorem to obtain linear observation and state transition equations).

The Jacobian is calculated using the package `ForwardDiff`

This function computes the first-order derivatives of the equilibrium conditions with
respect to states and controls, constructs the linear system, and solves it. It handles the
idiosyncratic part of the Jacobian efficiently by exploiting known derivatives (see
[`set_known_derivatives_distr!`](@ref)) and using automatic differentiation for the rest.

# Arguments

  - `sr`: steady-state structure (variable values, indexes, numerical parameters, ...).
  - `m_par`: model parameters.
  - `A`,`B`: derivative of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and `XPrime`
    [`A`]. Can be initialized as zero matrices.
  - `allow_approx_sol`: if `true` (default), the function will attempt to solve the
    linearized model even if the system is indeterminate (shifting the critical
    eigenvalues).
  - `ss_only`: if `true`, the function will only check if the steady state is a solution
    (i.e., if `Fsys` evaluates to zero at steady state) and return the residuals.

# Returns

  - `gx`: observation equations matrix (mapping states to controls).
  - `hx`: state transition equations matrix (mapping states to future states).
  - `alarm_LinearSolution`: `true` if the solution algorithm fails or is indeterminate.
  - `nk`: number of predetermined variables (states).
  - `A`,`B`: first derivatives of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`].
"""
function LinearSolution(
    sr,
    m_par,
    A::Array,
    B::Array;
    allow_approx_sol = true,
    ss_only = false,
)
    ## --------------------------------------------------------------------------
    ## Prepare elements used for uncompression
    ## --------------------------------------------------------------------------

    transform_elements =
        transformation_elements(sr, sr.n_par.model, sr.n_par.distribution_states)

    ## --------------------------------------------------------------------------
    ## Check whether Steady state solves the difference equation
    ## --------------------------------------------------------------------------

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

    if ss_only
        return FR
    end

    if sr.n_par.verbose
        @printf "Number of States and Controls: %d\n" length(F)
        @printf "Max error on Fsys: %.2e\n" maximum(abs.(FR[:]))
        if typeof(sr.n_par.model) != CompleteMarkets
            for field in propertynames(sr.indexes.distr)
                @printf "Max error of distribution %s in Fsys: %.2e\n" field maximum(
                    abs.(FR[getfield(sr.indexes.distr, field)]),
                )
            end
            for field in propertynames(sr.indexes.valueFunction)
                @printf "Max error of value function %s in Fsys: %.2e\n" field maximum(
                    abs.(FR[getfield(sr.indexes.valueFunction, field)]),
                )
            end
        end
    end

    ## --------------------------------------------------------------------------
    ## Calculate Jacobians of the Difference equation F
    ## --------------------------------------------------------------------------
    # BA = ForwardDiff.jacobian(
    #     x -> Fsys(
    #         x[1:length_X0],
    #         x[(length_X0 + 1):end],
    #         sr.XSS,
    #         m_par,
    #         sr.n_par,
    #         sr.indexes,
    #         sr.compressionIndexes,
    #         transform_elements,
    #     ),
    #     zeros(2 * length_X0),
    # )

    # B = BA[:, 1:length_X0]
    # A = BA[:, (length_X0 + 1):end]

    f(x) = Fsys(
        x[1:length_X0],
        x[(length_X0 + 1):end],
        sr.XSS,
        m_par,
        sr.n_par,
        sr.indexes,
        sr.compressionIndexes,
        transform_elements,
    )

    # with known derivatives
    # -----------

    function manip_some(x, indexes, lengthy)
        y = zeros(eltype(x), lengthy)
        y[indexes] = x
        return y
    end
    A = zeros(length_X0, length_X0)
    B = zeros(length_X0, length_X0)

    dist_indexes =
        vcat([getfield(sr.indexes.distr, d) for d in propertynames(sr.indexes.distr)]...)
    V_indexes = vcat(
        [
            getfield(sr.indexes.valueFunction, v) for
            v in propertynames(sr.indexes.valueFunction)
        ]...,
    )

    not_dist_indexes = setdiff(1:length_X0, dist_indexes)
    not_V_indexes = setdiff(1:length_X0, V_indexes)

    set_known_derivatives_distr!(
        A,
        dist_indexes,
        transform_elements,
        sr.indexes.distr,
        sr.n_par,
        sr.n_par.transition_type,
        sr.n_par.transf_CDF,
    )
    # Derivatives with respect to time t value functions are known (unit matrix)
    B[V_indexes, V_indexes] = I[1:length(V_indexes), 1:length(V_indexes)]

    A[:, not_dist_indexes] = ForwardDiff.jacobian(
        x -> f(manip_some(x, not_dist_indexes .+ length_X0, 2 * length_X0)),
        zeros(length(not_dist_indexes)),
    )
    B[:, not_V_indexes] = ForwardDiff.jacobian(
        x -> f(manip_some(x, not_V_indexes, 2 * length_X0)),
        zeros(length(not_V_indexes)),
    )

    ## --------------------------------------------------------------------------
    ## Solve the linearized model: Policy Functions and LOMs
    ## --------------------------------------------------------------------------
    gx, hx, alarm_LinearSolution, nk = SolveDiffEq(A, B, sr.n_par, allow_approx_sol)

    if sr.n_par.verbose
        @printf "State Space Solution Done\n"
    end

    return gx, hx, alarm_LinearSolution, nk, A, B
end

"""
    set_known_derivatives_distr!(A, dist_indexes, transform_elements, indexes,
                                 n_par, transition_type, transform_type)

Pre-fill known derivative blocks of the Jacobian matrix `A` (with respect to future states)
for the distributional part of the state vector. This avoids redundant automatic differentiation
and significantly speeds up linearization.

The function exploits the structure of the distribution transition equations: the Jacobian
with respect to future distributions follows directly from the state transition rules (Γ
matrices) and the distribution representation (copula, CDF, or representative agent).

Multiple methods handle different combinations of:

  - `indexes`: Type indicating distribution representation (`CopulaTwoAssetsIndexes`, `CopulaOneAssetIndexes`, `CDFIndexes`, `RepAgentIndexes`).
  - `transition_type`: `LinearTransition` or `NonLinearTransition`.
  - `transform_type`: `LinearTransformation` or `ParetoTransformation`.

# Arguments

  - `A::AbstractMatrix`: Jacobian matrix (filled in-place).
  - `dist_indexes::Vector`: Indices in the state vector corresponding to distributional variables.
  - `transform_elements::TransformationElements`: Pre-computed transition operators (Γ matrices).
  - `indexes`: Index struct specifying the distribution representation.
  - `n_par::NumericalParameters`: Numerical parameters (grid sizes, etc.).
  - `transition_type`: Either `LinearTransition` or `NonLinearTransition`.
  - `transform_type`: Either `LinearTransformation` or `ParetoTransformation`.

# Behavior

Sets `A[dist_indexes, dist_indexes]` to `-I` (negative identity) as the base, then modifies
specific blocks corresponding to each asset/income dimension using the transition matrices Γ.
For CDF-based states with linear transformation, handles both unit and shuffled derivatives
via cumsum logic. For Pareto transformation, uses only the base `-I` block.
"""
function set_known_derivatives_distr!(
    A,
    dist_indexes,
    transform_elements::TransformationElements,
    indexes::CopulaTwoAssetsIndexes,
    n_par::NumericalParameters,
    ::LinearTransition,
    ::Union{LinearTransformation,ParetoTransformation},
)
    # Derivatives with respect to time t+1 distributions are known (unit/shuffle mat)
    A[dist_indexes, dist_indexes] = -I[1:length(dist_indexes), 1:length(dist_indexes)]
    A[indexes.b, indexes.b] = -transform_elements.Γ[1][1:(end - 1), :]
    A[indexes.k, indexes.k] = -transform_elements.Γ[2][1:(end - 1), :]
    A[indexes.h, indexes.h] = -transform_elements.Γ[3][1:(end - 1), :]
end

function set_known_derivatives_distr!(
    A,
    dist_indexes,
    transform_elements::TransformationElements,
    indexes::CopulaOneAssetIndexes,
    n_par::NumericalParameters,
    ::LinearTransition,
    ::Union{LinearTransformation,ParetoTransformation},
)
    # Derivatives with respect to time t+1 distributions are known (unit/shuffle mat)
    A[dist_indexes, dist_indexes] = -I[1:length(dist_indexes), 1:length(dist_indexes)]
    A[indexes.b, indexes.b] = -transform_elements.Γ[1][1:(end - 1), :]
    A[indexes.h, indexes.h] = -transform_elements.Γ[2][1:(end - 1), :]
end

function set_known_derivatives_distr!(
    A,
    dist_indexes,
    transform_elements::TransformationElements,
    indexes::CopulaOneAssetIndexes,
    n_par::NumericalParameters,
    ::NonLinearTransition,
    ::Union{LinearTransformation,ParetoTransformation},
)
    # Derivatives with respect to time t+1 distributions are known (unit/shuffle mat)
    A[dist_indexes, dist_indexes] = -I[1:length(dist_indexes), 1:length(dist_indexes)]
    A[indexes.h, indexes.h] = -transform_elements.Γ[2][1:(end - 1), :]
end

function set_known_derivatives_distr!(
    A,
    dist_indexes,
    transform_elements::TransformationElements,
    indexes::CDFIndexes,
    n_par::NumericalParameters,
    ::TransitionType,
    ::LinearTransformation,
)
    @assert dist_indexes[1] == 1 "First index in dist_indexes should be 1 for CDFStates."
    # Derivatives with respect to time t+1 distributions are known (unit/shuffle mat)
    unit_indices = indexes_unit_derivatives(n_par.nb, n_par.nh)
    for ci in unit_indices
        A[ci] = -1.0
    end

    # The last value of a conditional is the income PDF, which is treated discretely and for
    # which the shuffle matrix is applied and then cumulated as object of interest is the
    # joint CDF.
    shuffled_indices = indexes_shuffled_derivatives(n_par.nb, n_par.nh)
    shuffle_values = cumsum(transform_elements.Γ[1] .* -1; dims = 1)[1:(end - 1), :][:]

    for (idx, ci) in enumerate(shuffled_indices)
        A[ci] += shuffle_values[idx]
    end
end

function set_known_derivatives_distr!(
    A,
    dist_indexes,
    transform_elements::TransformationElements,
    indexes::CDFIndexes,
    n_par::NumericalParameters,
    ::TransitionType,
    ::ParetoTransformation,
)
    A[dist_indexes, dist_indexes] = -I[1:length(dist_indexes), 1:length(dist_indexes)]
end

function set_known_derivatives_distr!(
    A,
    dist_indexes,
    transform_elements::TransformationElements,
    indexes::RepAgentIndexes,
    n_par::NumericalParameters,
    ::TransitionType,
    ::Union{LinearTransformation,ParetoTransformation},
)
    A[dist_indexes, dist_indexes] = -I[1:length(dist_indexes), 1:length(dist_indexes)]
    A[indexes.h, indexes.h] = -transform_elements.Γ[1][1:(end - 1), :]
end

"""
    indexes_unit_derivatives(nb, nh)

Get the Cartesian indices for unit derivatives in the distribution matrix with CDF as
states.

Relevant indices for the unit derivative follow from the perturbation of a state itself and
the indices following from the cumsum.

# Arguments

  - `nb`: Number of grid points for liquid assets.
  - `nh`: Number of grid points for human capital.

# Returns

  - `indices`: Vector of `CartesianIndex{2}` representing the locations in the matrix.
"""
function indexes_unit_derivatives(nb, nh)
    indices = []
    i_count = 1
    for h = 1:nh, i = 1:(nb - 1), j = 1:nh
        row = i + nb * (h - 1) + nb * (j - 1)
        col = i + nb * (j - 1)
        if 0 < row <= nb * nh
            push!(indices, CartesianIndex(row, col))
            i_count += 1
        end
    end
    return indices
end

"""
    indexes_shuffled_derivatives(nb, nh)

Get the Cartesian indices for the non-unit derivatives in the distribution matrix with CDF
as states.

Relevant indices are the end of each conditional (except the last one) and indices following
from the cumsum logic used in the shuffle matrix.

# Arguments

  - `nb`: Number of grid points for liquid assets.
  - `nh`: Number of grid points for human capital.

# Returns

  - `indices`: Vector of `CartesianIndex{2}` representing the locations in the matrix.
"""
function indexes_shuffled_derivatives(nb, nh)
    total_count = (nh - 1)^2
    indices = Vector{CartesianIndex{2}}(undef, total_count)
    i_count = 1
    for ih1 = 1:(nh - 1)
        for ih2 = 1:(nh - 1)
            indices[i_count] = CartesianIndex(ih2 * nb, ih1 * nb)
            i_count += 1
        end
    end
    return indices
end
