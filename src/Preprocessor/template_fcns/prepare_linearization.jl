#=

Template function for prepare_linearization.jl

Explanation: During the preprocessing step, `PreprocessInputs.jl` reads in this template
file and copies the content from `input_aggregate_steady_state.mod` into the code below at
the line marked with "# aggregate steady state marker". The code block is then written to
`prepare_linearization_generated.jl` in the `generated_fcns` directory.

=#

"""
    prepare_linearization(K, Wb, Wk, distr, n_par, m_par)

Given the stationary equilibrium of the household side, computed in
[`find_steadystate()`](@ref), this function performs several steps:

  - Step 1: compute the stationary equilibrium.
  - Step 2: perform the dimensionality reduction of the marginal value functions as well as
    the distribution.
  - Step 3: compute the aggregate steady state from `input_aggregate_steady_state.mod`.
  - Step 4: produce indexes to access the variables in the linearized model.
  - Step 5: return the results.

# Arguments

  - `K::Float64`: steady-state capital stock
  - `Wb::Array{Float64,3}`, `Wk::Array{Float64,3}`: steady-state marginal value functions
  - `distr::Array{Float64,3}`: steady-state distribution of idiosyncratic states
  - `n_par::NumericalParameters`,`m_par::ModelParameters`

# Returns

  - `XSS::Array{Float64,1}`, `XSSaggr::Array{Float64,1}`: steady state vectors produced by
    [`@writeXSS()`](@ref)
  - `indexes`, `indexes_aggr`: `struct`s for accessing `XSS`,`XSSaggr` by variable names,
    produced by [`@make_fn()`](@ref), [`@make_fnaggr()`](@ref)
  - `compressionIndexes::Array{Array{Int,1},1}`: indexes for compressed marginal value
    functions (`V_m` and `V_k`)
  - `n_par::NumericalParameters`, `m_par::ModelParameters`: updated parameters
  - `CDFSS`, `CDF_bSS`, `CDF_kSS`, `CDF_hSS`: cumulative distribution functions (joint and
    marginals)
  - `distrSS::Array{Float64,3}`: steady state distribution of idiosyncratic states, computed
    by [`Ksupply()`](@ref)
"""
function prepare_linearization(
    K::Float64,
    vf::ValueFunctions,
    distr::Array{Float64},
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    # Guarantee stability of the scope of the function
    KSS = copy(K)
    vfSS = copy(vf)
    distrSS = copy(distr)

    ## ------------------------------------------------------------------------------------
    ## Step 1: Evaluate stationary equilibrium to calculate steady state variable values
    ## ------------------------------------------------------------------------------------

    ## Aggregate part ---------------------------------------------------------------------

    # See module IncomesETC for details on the function compute_args_hh_prob_ss
    args_hh_prob = compute_args_hh_prob_ss(KSS, m_par, n_par)

    @read_args_hh_prob_ss()

    # Store the arguments in a named tuple, for checking consistency
    args_hh_prob_tuple = NamedTuple{Tuple(Symbol.(args_hh_prob_names))}(args_hh_prob)

    ## Incomes ----------------------------------------------------------------------------

    # Net incomes of households
    net_income, gross_income, eff_int = incomes(n_par, m_par, args_hh_prob)

    ## Idiosyncratic part -----------------------------------------------------------------

    # Solution to household problem in the stationary equilibrium, given args_hh_prob
    KSS, BSS, transition_matricesSS, pfSS, vfSS, distrSS =
        Ksupply(args_hh_prob, n_par, m_par, vfSS, distrSS, net_income, eff_int)
    vfSS.b .*= eff_int

    # Produce distribution structure with CDFSS and potentially marginals
    distrSS = set_distribution(
        pdf_to_cdf(distrSS),
        n_par.model,
        n_par.distribution_states,
        n_par.transition_type,
    )
    distrXSS_vec = vcat(vec.(struc_to_vec(distrSS))...)

    # Distributional summary statistics (for filling in the steady state vector)
    TOP10WshareSS, TOP10IshareSS, TOP10InetshareSS, GiniWSS, GiniCSS, sdlogySS =
        distrSummaries(distrSS, qSS, pfSS, n_par, net_income, gross_income, m_par)

    ## ------------------------------------------------------------------------------------
    ## Step 2: Dimensionality reduction
    ## ------------------------------------------------------------------------------------

    compressionIndexes = dimensionality_reduction(
        vfSS,
        transition_matricesSS,
        pfSS,
        args_hh_prob,
        m_par,
        n_par,
        n_par.transition_type,
        n_par.distribution_states,
    )

    # Apply log inverse transformation to value Functions
    for f in fieldnames(typeof(vfSS))
        setproperty!(vfSS, f, log.(invmutil(getfield(vfSS, f), m_par)))
    end
    vfSS_vec = vcat(vec.(struc_to_vec(vfSS))...)
    # Produce marginal distributions
    copula_marginal_b, copula_marginal_k, copula_marginal_h =
        produce_marginals(distrSS, n_par)

    @set! n_par.copula_marginal_b = copula_marginal_b
    @set! n_par.copula_marginal_k = copula_marginal_k
    @set! n_par.copula_marginal_h = copula_marginal_h

    ## ------------------------------------------------------------------------------------
    ## Step 3: Get the aggregate steady state (`input_aggregate_steady_state.mod`)
    ## ------------------------------------------------------------------------------------

    # DO NOT DELETE OR EDIT NEXT LINE! This is needed for parser.

    # aggregate steady state marker

    # Write steady state values into XSS vector
    @writeXSS

    ## ------------------------------------------------------------------------------------
    ## Step 4: Produce indexes to access the variables in the linearized model
    ## ------------------------------------------------------------------------------------

    # produce indexes to access XSS etc.
    indexes = produce_indexes(n_par, compressionIndexes[1], compressionIndexes[2])
    indexes_aggr = produce_indexes_aggr(n_par)

    n_par = set_npar_sizes(n_par, compressionIndexes)

    ## ------------------------------------------------------------------------------------
    ## Step 5: Check consistency
    ## ------------------------------------------------------------------------------------

    # Check that the steady state values in the aggregate model are consistent
    for (key, value) in pairs(args_hh_prob_tuple)
        if !isapprox(exp(XSS[getfield(indexes, Symbol(key, "SS"))]), value)
            @warn "Inconsistency detected for $key: expected $value, got $(exp(XSS[getfield(indexes, Symbol(key, "SS"))]))"
        end
    end

    ## ------------------------------------------------------------------------------------
    ## Step 6: Return results
    ## ------------------------------------------------------------------------------------

    return XSS, XSSaggr, indexes, indexes_aggr, compressionIndexes, n_par, m_par, distrSS
end

## ----------------------------------------------------------------------------------------
## Helper functions
## ----------------------------------------------------------------------------------------

"""
    copula_marg_equi_h(CDF_i, grid_i, nx)
"""
function copula_marg_equi_h(CDF_i, grid_i, nx)
    distr_i = [0.0; diff(CDF_i)]
    aux_marginal = collect(range(CDF_i[1]; stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 2:(nx - 1)
        equi(x1) = equishares(x1, x2, grid_i[1:(end - 1)], distr_i[1:(end - 1)], nx - 1)
        x2 = find_zero(equi, (1e-9, x2))
        aux_marginal[end - i] = x2
    end

    aux_marginal[end] = CDF_i[end]
    aux_marginal[1] = CDF_i[1]
    aux_marginal[end - 1] = CDF_i[end - 1]
    copula_marginal = copy(aux_marginal)
    jlast = nx - 1
    for i = (nx - 2):-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j
            j -= 1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

"""
    copula_marg_equi(CDF_i, grid_i, nx)
"""
function copula_marg_equi(CDF_i, grid_i, nx)
    distr_i = [0.0; diff(CDF_i)]
    aux_marginal = collect(range(CDF_i[1]; stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 1:(nx - 1)
        equi(x1) = equishares(x1, x2, grid_i, distr_i, nx)
        x2 = find_zero(equi, (eps(), x2))
        aux_marginal[end - i] = x2
    end

    aux_marginal[end] = CDF_i[end]
    aux_marginal[1] = CDF_i[1]
    copula_marginal = copy(aux_marginal)
    jlast = nx
    for i = (nx - 1):-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j
            j -= 1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

"""
    copula_marg_equi(CDF_i::AbstractVector, grid_i::AbstractVector, nx::Int; enforce_first_x::Int=1)

Sample from the distribution `distr_i` using equishared weights.

# Arguments

  - `distr_i::AbstractVector`: The input PDF representing the marginal distribution.
  - `grid_i::AbstractVector`: The grid corresponding to the PDF.
  - `nx::Int`: The number of grid points for the output marginal distribution.
  - `enforce_first_x::Int=1`: The number of grid points to enforce at the beginning of the
    marginal distribution.

# Returns

  - `copula_marginal::AbstractVector`: A strictly increasing marginal distribution for the
    copula.

# Notes

  - The function ensures that the resulting marginal distribution is strictly increasing.
  - The first `enforce_first_x` grid points are fixed to the corresponding values in the
    input grid.
  - The function uses a combination of equidistant and convex weighting methods to generate
    the marginal distribution.
"""
function copula_marg_equi_convex(CDF_i, grid_i, nx; enforce_first_x = 1)
    distr_i = [0.0; diff(CDF_i)]         # Marginal distribution (cdf) of liquid assets
    nx_free = nx - enforce_first_x
    aux_marginal =
        collect(range(CDF_i[enforce_first_x + 1]; stop = CDF_i[end], length = nx_free))
    aux_margin_unif = copy(aux_marginal)

    x2 = 1.0 - 1e-14
    for i = 1:(nx_free - 1)
        equi(x1) = equishares(x1, x2, grid_i, distr_i, nx_free)
        x2 = find_zero(equi, (1e-9, x2))
        aux_marginal[end - i] = x2
    end

    aux_marginal[end] = CDF_i[end]

    # produce convex weights
    if 1 < enforce_first_x
        λ = 0.001
    else
        λ = 0.1
    end
    weights = λ .^ collect(range(0; stop = 1, length = nx_free))
    aux_marginal = [
        weights[i] * aux_margin_unif[i] + (1 - weights[i]) * m for
        (i, m) in enumerate(aux_marginal)
    ] # convex combi
    copula_marginal = copy(aux_marginal)
    jlast = length(CDF_i) - 1
    for i = (nx_free - 1):-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        # ensure that every element in the middle of copula_marginal is unique!
        if jlast <= j
            j = max(1, jlast - 1)
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end

    copula_marginal =
        enforce_first_x > 0 ? [CDF_i[1:enforce_first_x]; copula_marginal] : copula_marginal

    # restructure if in the beginning, more than one element is CDF_i[1]
    ix_same = findall(copula_marginal .== CDF_i[1])
    len_same = length(ix_same)
    if 1 < len_same
        k = 0
        ix_last = length(CDF_i)
        while length(CDF_i) - 1 - ix_last < len_same
            k += 1
            ix_last = findall(CDF_i .== copula_marginal[end - k])[1]
        end
        cop_marg_new = copy(copula_marginal)
        cop_marg_new[1:(nx - len_same - k + 1)] = copula_marginal[len_same:(end - k)]
        cop_marg_new[(nx - len_same + 2 - k):(end - 1)] =
            CDF_i[(ix_last + 1):(ix_last + len_same + k - 2)]
        copula_marginal = cop_marg_new
    end
    @assert all(diff(copula_marginal) .> 0)
    return copula_marginal
end

"""
    equishares(x1, x2, grid_i, distr_i, nx)
"""
function equishares(x1, x2, grid_i, distr_i, nx)
    FN_Wshares = cumsum(grid_i .* distr_i) ./ sum(grid_i .* distr_i)
    Wshares = diff(mylinearinterpolate(cumsum(distr_i), FN_Wshares, [x1; x2]))
    dev_equi = Wshares .- 1.0 ./ nx

    return dev_equi
end

function produce_marginals(distrSS::CopulaTwoAssets, n_par::NumericalParameters)
    dt = typeof(distrSS)
    copula_marginal_b =
        copula_marg_equi(get_CDF(distrSS.b, dt), n_par.grid_b, n_par.nb_copula)
    copula_marginal_k =
        copula_marg_equi(get_CDF(distrSS.k, dt), n_par.grid_k, n_par.nk_copula)
    copula_marginal_h =
        copula_marg_equi_h(get_CDF(distrSS.h, dt), n_par.grid_h, n_par.nh_copula)
    return copula_marginal_b, copula_marginal_k, copula_marginal_h
end

function produce_marginals(distrSS::CopulaOneAsset, n_par::NumericalParameters)
    dt = typeof(distrSS)
    copula_marginal_b =
        copula_marg_equi(get_CDF(distrSS.b, dt), n_par.grid_b, n_par.nb_copula)
    # copula_marginal_b =
    #     copula_marg_equi_convex(get_CDF(distrSS.b, dt), n_par.grid_b, n_par.nb_copula)
    copula_marginal_k = [1.0]
    copula_marginal_h =
        copula_marg_equi_h(get_CDF(distrSS.h, dt), n_par.grid_h, n_par.nh_copula)
    isa(n_par.transition_type, NonLinearTransition) &&
        minimum(diff(copula_marginal_b)) .<= 0.0 &&
        @warn "Copula marginal for b not strictly monotone, which can cause issues in Fsys with NonLinearTransition."
    # isa(n_par.transition_type, NonLinearTransition) &&
    #     minimum(diff(copula_marginal_h)) .<= 0.0 &&
    #     @warn "Copula marginal for h not strictly monotone, which can cause issues in Fsys with NonLinearTransition."
    return copula_marginal_b, copula_marginal_k, copula_marginal_h
end

function produce_marginals(distrSS::Union{CDF,RepAgent}, n_par::NumericalParameters)
    return [1.0], [1.0], [1.0]
end

function dimensionality_reduction(
    vfSS::Union{ValueFunctionsOneAsset,ValueFunctionsTwoAssets},
    transition_matricesSS::TransitionMatrices,
    pfSS::PolicyFunctions,
    args_hh_prob::Vector,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::LinearTransition,
    distribution_states::CopulaStates,
)

    ## DCT coefficients for marginal value functions --------------------------------------

    # Identify the co-linear variables in arguments of household problem
    exclude_list = ["N", "Hprog", "Htilde"]
    include_list_idx = findall(x -> x ∉ exclude_list, args_hh_prob_names)

    compressionIndexesVf, _ = first_stage_reduction(
        vfSS,
        transition_matricesSS,
        pfSS,
        args_hh_prob,
        include_list_idx,
        n_par,
        m_par,
    )

    ## Indices for distribution perturbation:
    # Polynomials indices for copula perturbation and CDF indices for CDF perturbation
    compressionIndexesDistr = distr_indices(n_par, n_par.distribution_states)

    # Return container to store all retained coefficients in one array
    return [compressionIndexesVf, compressionIndexesDistr]
end

function dimensionality_reduction(
    vfSS::ValueFunctionsOneAsset,
    transition_matricesSS::TransitionMatrices,
    pfSS::PolicyFunctions,
    args_hh_prob::Vector,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::NonLinearTransition,
    distribution_states::CopulaStates,
)

    ## DCT coefficients for marginal value functions --------------------------------------
    compressionIndexesVf = select_DCT_indices(vfSS, n_par)

    ## Indices for distribution perturbation:
    # Polynomials indices for copula perturbation and CDF indices for CDF perturbation
    compressionIndexesDistr = distr_indices(n_par, n_par.distribution_states)

    # Return container to store all retained coefficients in one array
    return [compressionIndexesVf, compressionIndexesDistr]
end

function dimensionality_reduction(
    vfSS::Union{ValueFunctionsOneAsset,ValueFunctionsTwoAssets},
    transition_matricesSS::TransitionMatrices,
    pfSS::PolicyFunctions,
    args_hh_prob::Vector,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::TransitionType,
    distribution_states::CDFStates,
)

    # No reduction for value function:
    compressionIndexesVf = if isa(n_par.model, TwoAsset)
        [
            collect(1:(n_par.nb * n_par.nk * n_par.nh)),
            collect(1:(n_par.nb * n_par.nk * n_par.nh)),
        ]
    else
        [collect(1:(n_par.nb * n_par.nh))]
    end
    if isa(n_par.model, OneAsset)
        @assert n_par.nk == 1
    end
    compressionIndexesDistr = collect(1:(n_par.nb * n_par.nh * n_par.nk))
    return [compressionIndexesVf, compressionIndexesDistr]
end

function dimensionality_reduction(
    vfSS::ValueFunctionsCompleteMarkets,
    transition_matricesSS::TransitionMatricesCompleteMarkets,
    pfSS::PolicyFunctions,
    args_hh_prob::Vector,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::TransitionType,
    distribution_states::Union{CDFStates,CopulaStates},
)
    return [[[]], collect(1:(n_par.nh))]
end

function distr_indices(n_par::NumericalParameters, distribution_states::CopulaStates)
    SELECT = [
        (!((i == 1) & (j == 1)) & !((k == 1) & (j == 1)) & !((k == 1) & (i == 1))) for
        i = 1:(n_par.nb_copula), j = 1:(n_par.nk_copula), k = 1:(n_par.nh_copula)
    ]

    # return indices of selected coeffs
    return findall(SELECT[:])
end

function distr_indices(n_par::NumericalParameters, distribution_states::CDFStates)
    return collect(1:(n_par.nb * n_par.nh * n_par.nk))
end

ntotal(n_par, compressionIndexes, ::Union{OneAsset,TwoAsset}, ::CDFStates) =
    length(vcat(vcat(compressionIndexes...)...)) - 1 + n_par.naggr
ntotal(n_par, compressionIndexes, ::TwoAsset, ::CopulaStates) =
    length(vcat(vcat(compressionIndexes...)...)) +
    (n_par.nh + n_par.nb + n_par.nk - 3 + n_par.naggr)
ntotal(n_par, compressionIndexes, ::OneAsset, ::CopulaStates) =
    length(vcat(vcat(compressionIndexes...)...)) + (n_par.nh + n_par.nb - 2 + n_par.naggr)
ntotal(n_par, compressionIndexes, ::CompleteMarkets, ::Union{CDFStates,CopulaStates}) =
    length(vcat(vcat(compressionIndexes...)...)) - 1 + n_par.naggr

nstates(n_par, compressionIndexesDistr, ::Union{OneAsset,TwoAsset}, ::CDFStates) =
    length(compressionIndexesDistr) - 1 + n_par.naggrstates
nstates(n_par, compressionIndexesDistr, ::TwoAsset, ::CopulaStates) =
    n_par.nh + n_par.nk + n_par.nb - 3 + n_par.naggrstates + length(compressionIndexesDistr)
nstates(n_par, compressionIndexesDistr, ::OneAsset, ::CopulaStates) =
    n_par.nh + n_par.nb - 2 + n_par.naggrstates + length(compressionIndexesDistr)
nstates(
    n_par,
    compressionIndexesDistr,
    ::CompleteMarkets,
    ::Union{CDFStates,CopulaStates},
) = length(compressionIndexesDistr) - 1 + n_par.naggrstates

function set_npar_sizes(n_par, compressionIndexes)
    @set! n_par.ntotal =
        ntotal(n_par, compressionIndexes, n_par.model, n_par.distribution_states)
    @set! n_par.nstates =
        nstates(n_par, compressionIndexes[2], n_par.model, n_par.distribution_states)
    @set! n_par.ncontrols = length(vcat(compressionIndexes[1]...)) + n_par.naggrcontrols
    @set! n_par.LOMstate_save = zeros(n_par.nstates, n_par.nstates)
    @set! n_par.State2Control_save = zeros(n_par.ncontrols, n_par.nstates)
    @set! n_par.nstates_r = copy(n_par.nstates)
    @set! n_par.ncontrols_r = copy(n_par.ncontrols)
    @set! n_par.ntotal_r = copy(n_par.ntotal)
    @set! n_par.PRightStates = Diagonal(ones(n_par.nstates))
    @set! n_par.PRightAll = Diagonal(ones(n_par.ntotal))

    if n_par.n_agg_eqn != n_par.naggr - length(n_par.distr_names)
        @warn "Inconsistency in number of aggregate variables and equations!"
    end

    return n_par
end
