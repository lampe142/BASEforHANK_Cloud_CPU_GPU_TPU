"""
    find_steadystate(m_par; n_par_kwargs::NamedTuple = NamedTuple())

Find the stationary equilibrium capital stock as well as the associated value functions and
the stationary distribution of idiosyncratic states.

This function solves for the market clearing capital stock in the Aiyagari model with
idiosyncratic income risk. It uses [`CustomBrent()`](@ref) to find the root of the excess
capital demand function, which is defined in [`Kdiff()`](@ref). The solution is obtained
first on a coarse grid and then on the actual grid.

# Arguments

  - `m_par::ModelParameters`
  - `n_par_kwargs::NamedTuple=NamedTuple()`: Additional keyword arguments passed to
    [`NumericalParameters()`](@ref) when initializing the numerical parameters.

# Returns

  - `KSS`: Steady-state capital stock
  - `vfSS`: Value functions
  - `distrSS::Array{Float64,3}`: Steady-state distribution of idiosyncratic states
  - `n_par::NumericalParameters`, `m_par::ModelParameters`
"""
function find_steadystate(m_par; n_par_kwargs::NamedTuple = NamedTuple())
    if !isempty(n_par_kwargs)
        @info "Find steady state with additional n_par_settings: $(n_par_kwargs)"
    end

    ## ------------------------------------------------------------------------------------
    ## Step 0: Take care of complete markets case
    ## ------------------------------------------------------------------------------------

    n_par = NumericalParameters(; m_par = m_par, n_par_kwargs...)
    @assert !(isa(n_par.model, TwoAsset) && isa(n_par.transition_type, NonLinearTransition)) "Only OneAsset model with NonLinearTransition is supported currently."
    if typeof(n_par.model) == CompleteMarkets
        @info "Complete Markets Model: skip steady state search, use CompMarketsCapital function."
        @assert @isdefined(CompMarketsCapital) "Complete Markets Model requires CompMarketsCapital function."
        rSS = (1.0 .- m_par.β) ./ m_par.β  # complete markets interest rate
        KSS = CompMarketsCapital(rSS, m_par)
        distrSS = (n_par.Π^1000)[1, :]
        n_par = NumericalParameters(;
            m_par = m_par,
            naggrstates = length(state_names),
            naggrcontrols = length(control_names),
            aggr_names = aggr_names,
            distr_names = distr_names,
            n_par_kwargs...,
        )
        vfSS = ValueFunctionsCompleteMarkets(ones(eltype(n_par.mesh_b), 1) .* rSS)
        return KSS, vfSS, distrSS, n_par, m_par
    end

    ## ------------------------------------------------------------------------------------
    ## Step 1: Find the stationary equilibrium for coarse grid
    ## This step is just to get a good starting guess for the actual grid.
    ## ------------------------------------------------------------------------------------

    ## Income process and income grids ----------------------------------------------------

    # Read out numerical parameters for starting guess solution with reduced income grid.

    # Filter out nh, nb, nk from n_par_kwargs if they exist
    filtered_kwargs =
        NamedTuple(k => v for (k, v) in pairs(n_par_kwargs) if k ∉ (:nh, :nb, :nk))
    n_par = NumericalParameters(;
        m_par = m_par,
        ϵ = 1e-6,
        nh = n_par.nh_coarse,
        nk = n_par.nk_coarse,
        nb = n_par.nb_coarse,
        filtered_kwargs...,
    )
    @set! n_par.transition_type = LinearTransition()

    ## Capital stock guesses --------------------------------------------------------------

    if @isdefined(CompMarketsCapital)
        if n_par.verbose
            @printf "CompMarketsCapital function is defined, used for guesses.\n"
        end
        rKmin = n_par.rKmin_coarse
        rKmax = (1.0 .- m_par.β) ./ m_par.β - 0.0025 # complete markets interest rate
        Kmin = CompMarketsCapital(rKmax, m_par)
        Kmax = CompMarketsCapital(rKmin, m_par)
    else
        if n_par.verbose
            @printf "CompMarketsCapital function is not defined, using bounds from n_par.\n"
        end
        Kmin = n_par.Kmin_coarse
        Kmax = n_par.Kmax_coarse
    end

    @assert 0.0 < Kmin < Kmax < Inf "Invalid capital stock bounds."
    @assert !isnan(Kmin) && !isnan(Kmax) "Invalid capital stock bounds."

    if n_par.verbose
        @printf "Kmin: %f, Kmax: %f\n" Kmin Kmax
    end

    ## Excess demand function -------------------------------------------------------------

    # Define the excess demand function based on Kdiff from fcn_kdiff.jl, the keyword
    # arguments allow for a faster solution in CustomBrent because the results of the
    # previous iteration can be used as starting values.
    n_assets = isa(n_par.model, OneAsset) ? 1 : 2
    d(
        K,
        initialize::Bool = true,
        valueFunc_guess = [zeros(Float64, ntuple(_ -> 1, 3)) for _ = 1:n_assets],
        distr_guess = n_par.dist_guess,
    ) = Kdiff(
        K,
        n_par,
        m_par;
        initialize = initialize,
        valueFunc_guess,
        distr_guess = distr_guess,
    )

    ## Find equilibrium capital stock on coarse grid --------------------------------------

    if n_par.verbose
        @printf "Find capital stock, coarse income grid.\n"
    end
    BrentOut = CustomBrent(d, Kmin, Kmax)
    KSS = BrentOut[1]

    if n_par.verbose
        @printf "Capital stock, coarse income grid, is: %f\n" KSS
    end

    ## ------------------------------------------------------------------------------------
    ## Step 2: Find the stationary equilibrium for actual grid
    ## ------------------------------------------------------------------------------------

    ## Update numerical parameters --------------------------------------------------------

    # Read out numerical parameters, update to model equations.
    n_par = NumericalParameters(;
        m_par = m_par,
        naggrstates = length(state_names),
        naggrcontrols = length(control_names),
        aggr_names = aggr_names,
        distr_names = distr_names,
        n_par_kwargs...,
    )

    ## Find equilibrium capital stock on actual grid --------------------------------------

    if n_par.verbose
        @printf "Find capital stock, actual income grid.\n"
    end

    BrentOut = CustomBrent(
        d,
        KSS * (1 - n_par.search_range),
        KSS * (1 + n_par.search_range);
        tol = n_par.ϵ,
    )
    KSS = BrentOut[1]
    vfSS = BrentOut[3][2]
    vfSS = valFunc_from_vec(vfSS, n_par.model)
    distrSS = BrentOut[3][3] # Solver requires vector based outputs for updating

    if n_par.verbose
        @printf "Capital stock, actual income grid, is: %f\n" KSS
    end

    ## ------------------------------------------------------------------------------------
    ## Return results
    ## ------------------------------------------------------------------------------------

    return KSS, vfSS, distrSS, n_par, m_par
end
