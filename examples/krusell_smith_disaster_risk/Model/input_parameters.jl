@metadata prior nothing
@metadata label ""
@metadata latex_label L""
@metadata long_name ""

"""
    ModelParameters()

A structure to collect all model parameters, including calibrated values and prior
distributions for estimation.

# Overview

  - This struct is designed for macroeconomic models and includes parameters related to
    household preferences, income processes, technological factors, monetary policy, fiscal
    policy, and exogenous shocks.
  - The parameters are annotated with metadata such as names (both ASCII and LaTeX), prior
    distributions, and a boolean flag indicating whether they are estimated.
  - Uses the `Parameters`, `FieldMetadata`, and `Flatten` packages to facilitate parameter
    management.

# Fields

Each field follows the structure:

```julia
parameter::T = value | "ascii_name" | L"latex_name" | prior_distribution | estimated
```

  - `parameter`: Internal model parameter name.
  - `value`: Default numerical value.
  - `ascii_name`: Human-readable name used in output or logging.
  - `latex_name`: Corresponding LaTeX notation for use in reports and documentation.
  - `prior_distribution`: Prior distribution for Bayesian estimation (if applicable).
  - `estimated`: Boolean indicating whether the parameter is estimated.
"""
@label @long_name @latex_label @prior @flattenable @with_kw struct ModelParameters{T}

    # Household preference parameters
    ξ::T = 1.0 | "xi" | "risk aversion" | L"\xi" | _ | false
    γ::T = 1.0 | "gamma" | "inverse Frisch elasticity" | L"\gamma" | _ | false
    β::T = 0.98 | "beta" | "discount factor" | L"\beta" | _ | false
    λ::T =
        0.065 | "lambda" | "illiquid asset adjustment probability" | L"\lambda" | _ | false

    # Individual income process
    ρ_h::T = 0.98 | "rho" | "autocorrelation of income shocks" | L"\rho" | _ | false
    σ_h::T = 0.139 | "sigma" | "standard deviation of income shocks" | L"\sigma" | _ | false
    N::T = 1.0 | "N" | "labor supply" | L"N" | _ | false

    # Technological parameters
    α::T = 0.32 | "alpha" | "capital share" | L"\alpha" | _ | false
    δ_0::T = (0.07 + 0.016) / 4 | "delta" | "depreciation rate" | L"\delta" | _ | false
    # Further steady-state parameters
    Tlev::T =
        1.0 .+ 0.0 | "tau_lev" | "income tax rate level (gross)" | L"\tau^l" | _ | false
    Tprog::T =
        1.0 .+ 0.0 |
        "tau_pro" |
        "income tax rate progressivity (gross)" |
        L"\tau^p" |
        _ |
        false
    Tbar::T = 1.0 + 0.0 | "tau_bar" | "average tax rate (gross)" | L"\bar \tau" | _ | false
    Tc::T = 1.0 + 0.0 | "Tc" | "VAT rate (gross)" | L"T_c" | _ | false
    Hprog::T = 1.0 + 0.0 | "Hprog" | "labor supply adjustment" | L"H_{prog}" | _ | false
    RRB::T = (1.0 .^ 0.25) | "RB" | "real rate on bonds (gross)" | L"RRB" | _ | false
    Rbar::T = 0.0 | "Rbar" | "borrowing wedge" | L"\bar R" | _ | false
    q::T = 1.0 | "q" | "price of capital" | L"q" | _ | false
    Z::T = 1.0 | "Z" | "TFP" | L"Z" | _ | false
    σ::T = 1.0 | "sigma" | "income risk" | L"\sigma" | _ | false
    ψ::T = 0.0 | "psi" | "(1 - share) of capital in liquid assets" | L"\psi" | _ | false

    # fiscal policy
    scale_prog::Bool =
        false |
        "scale_prog" |
        "scaling of tax rate with tax base" |
        "scale_prog" |
        _ |
        false

    # exogeneous aggregate "shocks"

    ρ_Z::T = 0.75 | "rho_Z" | "autocorrelation of TFP shock" | L"\rho_Z" | _ | false
    σ_Z::T = 0.007 | "sigma_Z" | "standard deviation of TFP shock" | L"\sigma_Z" | _ | false
    τ_Z::T = 0.0 | "tau_Z" | "third moment^(1/3) of TFP shock" | L"\tau_Z" | _ | false
    ρ_delta::T =
        0.0 |
        "rho_D" |
        "Persistence of Capital destruction" |
        L"\rho_D" |
        Beta(beta_pars(0.5, 0.2^2)...) |
        true
    σ_delta::T =
        0.005 |
        "sigma_D" |
        "standard deviation of capital destruction" |
        L"\sigma_D" |
        InverseGamma(ig_pars(0.001, 0.02^2)...) |
        true
    τ_delta::T =
        0.012 |
        "tau_D" |
        "Third moment^(1/3) of capital destruction" |
        L"\tau_D" |
        _ |
        false
    ρ_beta::T =
        0.999 |
        "rho_beta" |
        "Persistence of beta shock" |
        L"\rho_{\beta}" |
        Beta(beta_pars(0.5, 0.2^2)...) |
        false
    σ_beta::T =
        0.05 |
        "sigma_beta" |
        "Standard deviation of beta shock" |
        L"\sigma_{\beta}" |
        _ |
        false
    τ_beta::T =
        0.0 | "tau_beta" | "Third moment^(1/3) of beta shock" | L"\tau_{\beta}" | _ | false
end

"""
    NumericalParameters()

Collect parameters for the numerical solution of the model in a `struct`.
"""
@with_kw struct NumericalParameters
    m_par::ModelParameters = ModelParameters()

    # If there are crashes in linear interpolations, then this often is because the grid is
    # not ideal or the interest rate is below 0. This can happen when the algorithm searches
    # for an equilibrium but does not need to be a serious mistake. To avoid the crashes,
    # set the following parameter to true, but consider adjusting the maximal gris
    # parameter.
    warn_egm::Bool = true

    # model we are solving
    model = OneAsset()
    transition_type = NonLinearTransition()
    distribution_states = CDFStates()
    entrepreneur::Bool = false
    GHH::Bool = false

    # regular grid
    nh::Int = 10
    nk::Int = 1
    nb::Int = 40

    # copula grid
    nh_copula::Int = nh # rule of thumb: divide nh, without entrepreneur, by two
    nk_copula::Int = 1 # rule of thumb: divide nk by twelve
    nb_copula::Int = 20 # rule of thumb: divide nb by twelve

    # coarse grid in find_steadystate
    nh_coarse::Int = 5
    nk_coarse::Int = 1
    nb_coarse::Int = 30

    # capital bounds for coarse grid in find_steadystate
    Kmin_coarse::Float64 = 5.0
    Kmax_coarse::Float64 = 30.0

    # other options for find_steadystate
    rKmin_coarse::Float64 = 0.0
    search_range::Float64 = 0.4

    # minimum and maximum values for grids
    kmin::Float64 = typeof(model) == TwoAsset ? 0.0 : 0.0
    kmax::Float64 = typeof(model) == TwoAsset ? 1000.0 : 0.0
    bmin::Float64 = typeof(model) == CompleteMarkets ? 0.0 : 0.0
    bmax::Float64 = typeof(model) == CompleteMarkets ? 0.0 : 200.0

    # precision of solution
    ϵ::Float64 = 1e-12

    sol_algo::Symbol = :schur # options: :schur (Klein's method), :lit (linear time iteration), :litx (linear time iteration with Howard improvement)
    verbose::Bool = true   # verbose model
    reduc_value::Float64 = 0.0   # Lost fraction of "energy" in the DCT compression for value functions
    reduc_marginal_value::Float64 = 0.0   # Lost fraction of "energy" in the DCT compression for value functions

    further_compress::Bool = true   # run model-reduction step based on MA(∞) representation
    further_compress_critC = 0.0  # critical value for eigenvalues for Value functions
    further_compress_critS = 1.0e-13      # critical value for eigenvalues for distribution

    # transformation of CDF, to ensure monotonicity and [0,1] bounds
    transf_CDF = LinearTransformation() # options: ParetoTransformation() or LinearTransformation()
    start_pareto_threshold::Float64 = eps() # threshold for starting pareto transformation
    end_pareto_threshold::Float64 = 1.0e-8 # threshold for ending pareto transformation

    # Parameters that will be overwritten in the code
    aggr_names::Array{String,1} = ["Something"] # Placeholder for names of aggregates
    distr_names::Array{String,1} = ["Something"] # Placeholder for names of distributions

    naggrstates::Int = 16 # (placeholder for the) number of aggregate states
    naggrcontrols::Int = 16 # (placeholder for the) number of aggregate controls
    nstates::Int = nh + nk + nb + naggrstates - 3 # (placeholder for the) number of states + controls in total
    ncontrols::Int = 16 # (placeholder for the) number of controls in total
    ntotal::Int = nstates + ncontrols     # (placeholder for the) number of states+ controls in total
    n_agg_eqn::Int = nstates + ncontrols     # (placeholder for the) number of aggregate equations
    naggr::Int = length(aggr_names)     # (placeholder for the) number of aggregate states + controls
    ntotal_r::Int = nstates + ncontrols# (placeholder for the) number of states + controls in total after reduction
    nstates_r::Int = nstates# (placeholder for the) number of states in total after reduction
    ncontrols_r::Int = ncontrols# (placeholder for the) number of controls in total after reduction

    PRightStates::AbstractMatrix = Diagonal(ones(nstates)) # (placeholder for the) Matrix used for second stage reduction (states only)
    PRightAll::AbstractMatrix = Diagonal(ones(ntotal))  # (placeholder for the) Matrix used for second stage reduction

    # Transition matrix and grid for income in steady state (worker - entrepreneur)
    grid_h::Array{Float64,1} = if entrepreneur
        [
            exp.(Tauchen(m_par.ρ_h, nh - 1)[1] .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h .^ 2))
            (m_par.ζ .+ m_par.ι) / m_par.ζ
        ]
    else
        exp.(Tauchen(m_par.ρ_h, nh)[1] .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h .^ 2))
    end
    Π::Matrix{Float64} = if entrepreneur
        [
            Tauchen(m_par.ρ_h, nh - 1)[2].*(1.0 .- m_par.ζ) m_par.ζ.*ones(nh - 1)
            m_par.ι ./ (nh - 1)*ones(1, nh - 1) 1.0 .- m_par.ι
        ]
    else
        Tauchen(m_par.ρ_h, nh)[2]
    end
    # bounds of income bins (except entrepreneur)
    bounds_h::Array{Float64,1} = if entrepreneur
        Tauchen(m_par.ρ_h, nh - 1)[3]
    else
        Tauchen(m_par.ρ_h, nh)[3]
    end

    # stationary equilibrium average human capital
    Htilde::Float64 = if entrepreneur
        ((Π^1000)[1, 1:(end - 1)]' * grid_h[1:(end - 1)])
    else
        ((Π^1000)[1, :]' * grid_h)
    end
    # stationary equilibrium fraction workers
    frac_workers::Float64 = if entrepreneur
        (1.0 / (1.0 - (Π^1000)[1, end]))
    else
        1.0
    end

    # initial gues for stationary distribution (needed if iterative procedure is used)
    dist_guess::Array{Float64} = if isa(model, TwoAsset)
        ones(nb, nk, nh) / (nb * nk * nh)
    else
        ones(nb, nh) / (nb * nh)
    end

    grid_type = :sqrt

    # grid illiquid assets:
    grid_k::Array{Float64,1} = if grid_type == :log
        exp.(range(log(kmin + 1.0); stop = log(kmax + 1.0), length = nk)) .- 1.0
    elseif grid_type == :sqrt
        nk > 1 ? (range(0; stop = sqrt(kmax - kmin), length = nk)) .^ 2 .+ kmin : [1.0]
    else
        error("Unknown grid type")
    end

    grid_b::Array{Float64,1} = if grid_type == :log
        exp.(range(log(bmin + 1.0); stop = log(bmax + 1.0), length = nb)) .- 1.0
    elseif grid_type == :sqrt
        nb > 1 ? (range(0; stop = sqrt(bmax - bmin + 1.0), length = nb)) .^ 2 .+ bmin : [1.0]
    else
        error("Unknown grid type")
    end

    # meshes for income, liquid and illiquid assets
    mesh_h::AbstractArray{Float64} =
        isa(model, TwoAsset) ? repeat(reshape(grid_h, (1, 1, nh)); outer = (nb, nk, 1)) :
        repeat(reshape(grid_h, (1, nh)); outer = (nb, 1))
    mesh_b::AbstractArray{Float64} =
        isa(model, TwoAsset) ? repeat(reshape(grid_b, (nb, 1, 1)); outer = (1, nk, nh)) :
        repeat(reshape(grid_b, (nb, 1)); outer = (1, nh))
    mesh_k::AbstractArray{Float64} =
        isa(model, TwoAsset) ? repeat(reshape(grid_k, (1, nk, 1)); outer = (nb, 1, nh)) :
        repeat(reshape(grid_k, (nk, 1)); outer = (1, nh))

    # grid for copula marginal distributions
    copula_marginal_b::Array{Float64,1} =
        nb == 1 ? [1.0] : collect(range(0.0; stop = 1.0, length = nb_copula))
    copula_marginal_k::Array{Float64,1} =
        nk == 1 ? [1.0] : collect(range(0.0; stop = 1.0, length = nk_copula))
    copula_marginal_h::Array{Float64,1} =
        nh == 1 ? [1.0] : collect(range(0.0; stop = 1.0, length = nh_copula))

    # Storage for linearization results
    LOMstate_save::Array{Float64,2} = zeros(nstates, nstates)
    State2Control_save::Array{Float64,2} = zeros(ncontrols, nstates)
end

"""
    EstimationSettings()

Collect settings for the estimation of the model parameters in a `struct`.

Use package `Parameters` to provide initial values. Input and output file names are stored
in the fields `mode_start_file`, `data_file`, `save_mode_file` and `save_posterior_file`.
"""
@with_kw struct EstimationSettings
    shock_names::Array{Symbol,1} = shock_names # set in Model/input_aggregate_names.jl
    observed_vars_input::Array{Symbol,1} = [
        :Ygrowth,
        :Igrowth,
        :Cgrowth,
        :N,
        :wgrowth,
        :RB,
        :π,
        :TOP10Wshare,
        :TOP10Ishare,
        :τprog,
        :σ,
    ]

    nobservables = length(observed_vars_input)

    data_rename::Dict{Symbol,Symbol} = Dict(
        :pi => :π,
        :sigma2 => :σ,
        :tauprog => :τprog,
        :w90share => :TOP10Wshare,
        :I90share => :TOP10Ishare,
    )

    me_treatment::Symbol = :unbounded
    me_std_cutoff::Float64 = 0.2

    meas_error_input::Array{Symbol,1} = [:TOP10Wshare, :TOP10Ishare]
    meas_error_distr::Array{InverseGamma{Float64},1} =
        [InverseGamma(ig_pars(0.01, 0.01^2)...), InverseGamma(ig_pars(0.01, 0.01^2)...)]

    # Leave empty to start with prior mode
    mode_start_file::String = ""

    data_file::String = ""
    save_mode_file::String = ""
    save_posterior_file::String = ""

    estimate_model::Bool = true

    max_iter_mode::Int = 3
    optimizer::Optim.AbstractOptimizer = NelderMead()
    compute_hessian::Bool = false    # true: computes Hessian at posterior mode; false: sets Hessian to identity matrix
    f_tol::Float64 = 1.0e-4
    x_tol::Float64 = 1.0e-4

    multi_chain_init::Bool = false
    ndraws::Int = 400
    burnin::Int = 100
    mhscale::Float64 = 0.00015
    debug_print::Bool = true
    seed::Int = 778187
end
