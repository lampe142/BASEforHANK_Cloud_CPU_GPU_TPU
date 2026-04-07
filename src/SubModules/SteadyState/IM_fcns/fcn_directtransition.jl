
# Transition with Lottery on PDF
get_dInit(distr::Union{CopulaOneAsset,CopulaTwoAssets}, transition_type::LinearTransition) =
    cdf_to_pdf(distr.COP)
get_dInit(distr::CDF, transition_type::LinearTransition) = cdf_to_pdf(distr.CDF)
# Transition with DEGM on conditional CDFs
get_dInit(
    distr::Union{CopulaOneAsset,CopulaTwoAssets},
    transition_type::NonLinearTransition,
) = hcat(distr.COP[:, 1], diff(distr.COP; dims = 2))
get_dInit(distr::CDF, transition_type::NonLinearTransition) =
    hcat(distr.CDF[:, 1], diff(distr.CDF; dims = 2))

init_dPrime(distr::Union{CopulaOneAsset,CopulaTwoAssets}) =
    zeros(eltype(distr.COP), size(distr.COP))
init_dPrime(distr::CDF) = zeros(eltype(distr.CDF), size(distr.CDF))
get_CDF(dPrime::AbstractArray, transition_type::LinearTransition) = pdf_to_cdf(dPrime) # Lottery returns PDF
get_CDF(dPrime::AbstractArray, transition_type::NonLinearTransition) =
    cumsum(dPrime; dims = 2) # Return CDF from conditional CDFs DEGM returns conditional CDFs
"""
    DirectTransition(
        b_a_star::Array,
        b_n_star::Array,
        k_a_star::Array,
        distr::Array,
        λ,
        Π::Array,
        n_par,
    )

Function to calculate next periods distribution of households given todays policy functions
and the current distribution.

The function calls the in-place version [`DirectTransition!`](@ref) to update the
distribution. For details on the implementation see that function and subsection
'Aggregation via non-stochastic simulations in ['ComputationalNotes.md'](ComputationalNotes.md) for further details.

# Arguments

  - `b_a_star::Array`: Liquid asset policy function for adjustment case
  - `b_n_star::Array`: Liquid asset policy function for non-adjustment case
  - `k_a_star::Array`: Illiquid asset policy function for adjustment case
  - `distr::Array`: Current distribution of households (as CDF)
  - `λ::Float64`: Probability of adjustment
  - `Π::Array`: Transition matrix for idiosyncratic productivity states
  - `n_par::NumericalParameters`: Struct holding numerical parameters

# Returns

  - `dPrime`: Updated distribution of households as CDF
"""
function DirectTransition(
    pf::PolicyFunctions,
    distr::Union{CDF,CopulaOneAsset,CopulaTwoAssets},
    λ,
    Π::Array,
    n_par,
)
    dPrime = init_dPrime(distr)
    DirectTransition!(
        dPrime,
        pf,
        get_dInit(distr, n_par.transition_type),
        λ,
        Π,
        n_par,
        n_par.transition_type,
    )
    return set_distribution(
        get_CDF(dPrime, n_par.transition_type),
        n_par.model,
        n_par.distribution_states,
        n_par.transition_type,
    )
end

function DirectTransition(pf::PolicyFunctions, distr::RepAgent, λ, Π::Array, n_par)
    return distr # no transition for representative agent
end

"""
    DirectTransition!(
        dPrime,
        b_a_star::Array,
        b_n_star::Array,
        k_a_star::Array,
        distr::Array,
        λ,
        Π::Array,
        n_par,
        model::Union{OneAsset, TwoAsset, CompleteMarkets},
    )

Function to calculate next periods distribution of households given todays policy functions
and the current distribution. The function updates the distribution in place.

The function is used in `FSYS.jl` function to update the distribution of households given
the policy functions and the current distribution. For details on the implementation see
subsection 'Aggregation via non-stochastic simulations in ['Computational
Notes.md'](ComputationalNotes.md) for further details.

# Arguments

  - `dPrime::Array`: To be updated distribution of households
  - `b_a_star::Array`: Liquid asset policy function for adjustment case
  - `b_n_star::Array`: Liquid asset policy function for non-adjustment case
  - `k_a_star::Array`: Illiquid asset policy function for adjustment case
  - `distr::Array`: Current distribution of households
  - `λ::Float64`: Probability of adjustment
  - `Π::Array`: Transition matrix for idiosyncratic productivity states
  - `n_par::NumericalParameters`: Struct holding numerical parameters
  - `model`: Model type, either `CompleteMarkets`, `OneAsset`, or `TwoAsset`

# Returns

  - `dPrime`: Updated distribution of households
"""
function DirectTransition!(
    dPrime,
    pf::PolicyFunctionsTwoAssets,
    distr::Array,
    λ,
    Π::Array,
    n_par,
    transition_type::LinearTransition,
)
    # Create linear interpolation weights from policy functions for both cases
    j_k_a, ω_right_k_a = MakeWeightsLight(pf.k_a_star, n_par.grid_k)
    j_b_a, ω_right_b_a = MakeWeightsLight(pf.b_a_star, n_par.grid_b)
    j_b_n, ω_right_b_n = MakeWeightsLight(pf.b_n_star, n_par.grid_b)

    # Setup the blockindex necessary for linear indexing over productivity state tomorrow
    # Below we will use this to calculate the target index of the policy functions for the
    # transition matrix. We need to add the blockindex to the linear index of the policy
    # function to get the correct target index associated with todays choices as function of
    # the state space and tomorrows productivity state
    blockindex = (0:(n_par.nh - 1)) * n_par.nk * n_par.nb

    # Loop over all states of the current period and calculate the updated distribution
    @inbounds begin
        for hh = 1:(n_par.nh) # all current productivity states
            for kk = 1:(n_par.nk) # all current illiquid asset states
                for bb = 1:(n_par.nb) # all current liquid asset states

                    # current mass of households in this state
                    dd = distr[bb, kk, hh]

                    # linear index of the policy functions for the current state
                    j_adj = (j_b_a[bb, kk, hh] .+ (j_k_a[bb, kk, hh] .- 1) .* n_par.nb)
                    j_non = (j_b_n[bb, kk, hh] .+ (kk - 1) .* n_par.nb)

                    # future mass for the liquid assets of non adjusters
                    ω = ω_right_b_n[bb, kk, hh]
                    d_L_n = (1.0 .- λ) .* (dd .* (1.0 .- ω))
                    d_R_n = (1.0 .- λ) .* (dd .* ω)

                    # future mass for the illiquid assets of adjusters
                    ω = ω_right_k_a[bb, kk, hh]
                    d_L_k_a = λ .* (dd .* (1.0 .- ω))
                    d_R_k_a = λ .* (dd .* ω)

                    # future mass for both assets of adjusters
                    ω = ω_right_b_a[bb, kk, hh]
                    d_LL_a = (d_L_k_a .* (1.0 .- ω))
                    d_LR_a = (d_L_k_a .* ω)
                    d_RL_a = (d_R_k_a .* (1.0 .- ω))
                    d_RR_a = (d_R_k_a .* ω)

                    # transitions to future productivity states
                    for hh_prime = 1:(n_par.nh)
                        # extract the probability of transitioning from the current
                        # productivity state to the future productivity state
                        pp = Π[hh, hh_prime]

                        # linear index of the policy functions for the future state
                        j_a = j_adj .+ blockindex[hh_prime]
                        j_n = j_non .+ blockindex[hh_prime]

                        # update the distribution of households by adding the weighted mass
                        # of households that will transition to the future state
                        # corresponding to the gammas in the transition matrix
                        dPrime[j_a] += pp .* d_LL_a
                        dPrime[j_a + 1] += pp .* d_LR_a
                        dPrime[j_a + n_par.nb] += pp .* d_RL_a
                        dPrime[j_a + n_par.nb + 1] += pp .* d_RR_a
                        dPrime[j_n] += pp .* d_L_n
                        dPrime[j_n + 1] += pp .* d_R_n
                    end
                end
            end
        end
    end
end

function DirectTransition!(
    dPrime,
    pf::PolicyFunctionsTwoAssets,
    distr::Array,
    λ,
    Π::Array,
    n_par,
    transition_type::NonLinearTransition,
)
    error("NonLinearTransition not implemented for TwoAsset model yet.")
end

function DirectTransition!(
    dPrime,
    pf::PolicyFunctionsOneAsset,
    distr::Array,
    λ,
    Π::Array,
    n_par,
    transition_type::LinearTransition,
)
    # Create linear interpolation weights from policy functions for both cases
    j_b_n, ω_right_b_n = MakeWeightsLight(pf.b_n_star, n_par.grid_b)

    # Setup the blockindex necessary for linear indexing over productivity state tomorrow
    # Below we will use this to calculate the target index of the policy functions for the
    # transition matrix. We need to add the blockindex to the linear index of the policy
    # function to get the correct target index associated with todays choices as function of
    # the state space and tomorrows productivity state
    blockindex = (0:(n_par.nh - 1)) * n_par.nb

    # Loop over all states of the current period and calculate the updated distribution
    @inbounds begin
        for hh = 1:(n_par.nh) # all current productivity states
            for bb = 1:(n_par.nb) # all current liquid asset states

                # current mass of households in this state
                dd = distr[bb, hh]

                # linear index of the policy functions for the current state
                j_non = j_b_n[bb, hh]

                # future mass for the liquid assets of non adjusters
                ω = ω_right_b_n[bb, hh]
                d_L_n = (dd .* (1.0 .- ω))
                d_R_n = (dd .* ω)

                # transitions to future productivity states
                for hh_prime = 1:(n_par.nh)
                    # extract the probability of transitioning from the current
                    # productivity state to the future productivity state
                    pp = Π[hh, hh_prime]

                    # linear index of the policy functions for the future state
                    j_n = j_non .+ blockindex[hh_prime]

                    # update the distribution of households by adding the weighted mass
                    # of households that will transition to the future state
                    # corresponding to the gammas in the transition matrix
                    dPrime[j_n] += pp .* d_L_n
                    dPrime[j_n + 1] += pp .* d_R_n
                end
            end
        end
    end
end

function DirectTransition!(
    dPrime,
    pf::PolicyFunctionsCompleteMarkets,
    distr::Array,
    λ,
    Π::Array,
    n_par,
    transition_type::TransitionType,
)
    # don't do anything
end

"""
    DirectTransition_Splines!(
        b_prime_grid::AbstractArray,
        cdf_initial::AbstractArray,
        n_par::NumericalParameters,
        bbar::Array
    )

    Direct transition of the savings cdf from one period to the next.
        Transition is done using monotonic spline interpolation to bring the next periods cdf's
        back to the reference grid.
        Logic: Given assets in t (on-grid) and an the income shock realization, the decision
        of next periods assets is deterministic and thus the probability mass move from the
        on grid to the off-grid values. Using the spline interpolation the cdf is evaluated at
        the fix asset grid.

    # Arguments
    - `cdf_prime_on_grid::AbstractArray`: Next periods CDF on fixed asset grid.
    - `b_prime_grid::AbstractArray`: Savings function defined on the fixed asset and income grid.
    - `cdf_initial::AbstractArray`: CDF over fixed assets grid for each income realization.
    - `Π::Array`: Stochastic transition matrix.
    - `n_par::NumericalParameters`
    - `m_par::ModelParameters`
"""
function DirectTransition!(
    cdf_prime_on_grid,
    pf::PolicyFunctionsOneAsset,
    cdf_initial::Array,
    λ,
    Π::Array,
    n_par::NumericalParameters,
    transition_type::NonLinearTransition,
)

    # 1. Map cdf back to fixed asset grid.
    @inbounds for i_h = 1:(n_par.nh)
        b_prime_given_h = view(pf.b_n_star, :, i_h)
        cdf_prime_given_h = cdf_initial[:, i_h]
        DirectTransition_Cond_Splines!(
            cdf_prime_given_h,
            b_prime_given_h,
            n_par,
            pf.b_tilde_n[1, i_h],
        )
        cdf_prime_on_grid[:, i_h] .= cdf_prime_given_h
    end
    # 2. Build expectation of cdf over income states
    cdf_prime_on_grid .= cdf_prime_on_grid * Π
end

"""
    DirectTransition_Cond_Splines!(cdf_prime_given_h, b_prime_given_h, n_par, bbar; ensure_top_monotonicity=true)

Map CDF conditional on one income state to the fixed asset grid using monotonic spline interpolation.

# Arguments

  - `cdf_prime_given_h::AbstractArray`: CDF of savings for a given income state.
  - `b_prime_given_h::AbstractArray`: Savings function for a given income state.
  - `n_par::NumericalParameters`: Numerical parameters of the model.
  - `bbar::Number`: Asset value that maps to the constraint.
  - `ensure_top_monotonicity`: If `true`, ensures monotonicity at the top of the CDF.

# Behavior

  - Maps the savings CDF `cdf_prime_given_h` back to the fixed asset grid using monotonic spline interpolation.
  - Ensures monotonicity of the CDF and adjusts for boundary conditions.
"""
function DirectTransition_Cond_Splines!(
    cdf_prime_given_h::AbstractArray,
    b_prime_given_h::AbstractArray,
    n_par::NumericalParameters,
    bbar::Number;
)
    # 1. Specify mapping from assets to cdf with monotonic PCIHP interpolation

    # Find CDF values at bbar (asset level that maps to constraint)
    idx_bbar = findlast(n_par.grid_b .<= bbar)
    if isnothing(idx_bbar) # i.e. bbar < 0
        idx_bbar = 1
        CDF_at_bbar = cdf_prime_given_h[1]
    else
        b_to_cdf_splines = Interpolator(n_par.grid_b, cdf_prime_given_h)
        CDF_at_bbar = b_to_cdf_splines(bbar)
    end

    # Find CDF where maximum CDF is reached to ensure strict monotonicity
    b_at_max_cdf = b_prime_given_h[end]
    idx_last_increasing_cdf = findlast(diff(cdf_prime_given_h) .> 0.0)
    if idx_last_increasing_cdf !== nothing
        b_at_max_cdf = b_prime_given_h[idx_last_increasing_cdf + 1]
    end

    b_prime_given_h_valid = [n_par.bmin; b_prime_given_h[(idx_bbar + 1):end]]
    cdf_prime_given_h_valid = [CDF_at_bbar; cdf_prime_given_h[(idx_bbar + 1):end]]

    @assert all(diff(b_prime_given_h_valid) .> 0) "b_prime_given_h_valid is not strictly increasing: $(b_prime_given_h_valid), $(b_prime_given_h)"

    # Interpolation only for unique and increasing values
    b_to_cdf_spline = Interpolator(b_prime_given_h_valid, cdf_prime_given_h_valid)

    # Extrapolation for values below (-> CDF = 0) and above observed m_primes (-> CDF = max)
    function b_to_cdf_spline_extr!(cdf_values::AbstractVector, b::AbstractVector)
        idx1 = findlast(b .< b_prime_given_h[1])
        if idx1 !== nothing
            cdf_values[1:idx1] .= 0.0
        else
            idx1 = 0
        end
        idx2 = findfirst(b .> min(b_at_max_cdf, b_prime_given_h[end]))
        if idx2 !== nothing
            cdf_values[idx2:end] .= 1.0 * cdf_prime_given_h[end]
        else
            idx2 = length(b) + 1
        end
        cdf_values[(idx1 + 1):(idx2 - 1)] .= b_to_cdf_spline.(b[(idx1 + 1):(idx2 - 1)])
    end

    # 2. Evaluate CDF at fixed grid
    cdfend = copy(cdf_prime_given_h[end])
    b_to_cdf_spline_extr!(cdf_prime_given_h, n_par.grid_b)
    # Ensure CDF at the end of the CDF is perceived as maximum
    cdf_prime_given_h .= min.(cdf_prime_given_h, cdfend)
    cdf_prime_given_h[end] = cdfend
end
