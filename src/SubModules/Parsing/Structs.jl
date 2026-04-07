"""
    @make_struct_aggr(struct_name)

Generate a lightweight `struct` named `struct_name` that contains integer index fields for
each name in the global `aggr_names` vector. For every aggregate name this macro creates two
integer fields: one named `<name>SS` for the steady-state index and one named `<name>` for
the deviation index.

Notes

  - Relies on a module-level `aggr_names` variable (array of names).
"""
macro make_struct_aggr(struct_name)
    a_names = Symbol.(aggr_names)
    n_states = length(a_names)

    fields_states = [:($(a_names[i])::Int) for i = 1:n_states]
    fieldsSS_states = [:($(Symbol(a_names[i], "SS"))::Int) for i = 1:n_states]

    esc(quote
        struct $struct_name
            $(fieldsSS_states...)
            $(fields_states...)
        end
    end)
end

"""
    @make_struct(struct_name)

Generate a `struct` named `struct_name` that contains index fields for state and control
variables and associated distribution/value-function indexes. For each `state_names` element
the macro creates `<name>SS` and `<name>` integer fields; for each `control_names` element
it creates `<name>SS` and `<name>` integer fields. The resulting struct also includes fields
`distrSS`, `valueFunctionSS`, `distr`, and `valueFunction` to hold index containers.

Notes

  - Relies on module-level `state_names` and `control_names` variables.
"""
macro make_struct(struct_name)
    s_names = Symbol.(state_names)
    n_states = length(s_names)
    c_names = Symbol.(control_names)
    n_controls = length(c_names)

    fields_states = [:($(s_names[i])::Int) for i = 1:n_states]
    fieldsSS_states = [:($(Symbol(s_names[i], "SS"))::Int) for i = 1:n_states]
    fields_controls = [:($(c_names[i])::Int) for i = 1:n_controls]
    fieldsSS_controls = [:($(Symbol(c_names[i], "SS"))::Int) for i = 1:n_controls]

    esc(quote
        struct $struct_name
            distrSS::DistributionIndexes
            $(fieldsSS_states...)
            valueFunctionSS::ValueFunctionIndexes
            $(fieldsSS_controls...)
            distr::DistributionIndexes
            $(fields_states...)
            valueFunction::ValueFunctionIndexes
            $(fields_controls...)
        end
    end)
end

"""
    SteadyStateStruct

Container for steady-state solution components.

Fields

  - `KSS`: steady-state aggregate capital (or placeholder type).
  - `vfSS`: value functions at steady state (`ValueFunctions`).
  - `distrSS`: steady-state distribution object (type varies by model).
  - `n_par`: numeric parameters related to solution dimensions.
"""
struct SteadyStateStruct
    KSS::Any
    vfSS::ValueFunctions
    distrSS::Any
    n_par::Any
end

"""
    SteadyResults

Holds outputs produced when computing steady-state results.

Fields

  - `XSS`: full steady-state variable collection.
  - `XSSaggr`: aggregate steady-state variables.
  - `indexes`, `indexes_r`, `indexes_aggr`: index maps for states/controls.
  - `compressionIndexes`: indexes used when compressing state space.
  - `n_par`, `m_par`: numeric parameters and model dimensions.
  - `distrSS`: steady-state distribution values (`DistributionValues`).
  - `state_names`, `control_names`: lists of state/control variable names.
"""
struct SteadyResults
    XSS::Any
    XSSaggr::Any
    indexes::Any
    indexes_r::Any
    indexes_aggr::Any
    compressionIndexes::Any
    n_par::Any
    m_par::Any
    distrSS::DistributionValues
    state_names::Any
    control_names::Any
end

"""
    LinearResults

Results from linearization and solution of the model's law of motion.

Fields

  - `State2Control`: mapping from states to controls (matrix or container).
  - `LOMstate`: linear law-of-motion state representation.
  - `A`, `B`: solution matrices.
  - `SolutionError`: diagnostic information about solution errors.
  - `nk`: numeric dimensions for state/control vectors.
"""
struct LinearResults
    State2Control::Any
    LOMstate::Any
    A::Any
    B::Any
    SolutionError::Any
    nk::Any
end

"""
    EstimResults

Container for estimation outputs and diagnostics.

Fields

  - `par_final`, `hessian_final`: estimated parameters and Hessian.
  - `meas_error`, `meas_error_std`: measurement error estimates.
  - `parnames`: names of estimated parameters.
  - `Data`, `Data_missing`: datasets used in estimation.
  - `IRFtargets`, `IRFserrors`: IRF-related targets and standard errors.
  - `H_sel`: selection matrix or related object.
  - `priors`: prior distributions used in estimation.
"""
struct EstimResults
    par_final::Any
    hessian_final::Any
    meas_error::Any
    meas_error_std::Any
    parnames::Any
    Data::Any
    Data_missing::Any
    IRFtargets::Any
    IRFserrors::Any
    H_sel::Any
    priors::Any
end

"""
    ValueFunctionsOneAsset

Value function container for the one-asset model.

Stores arrays representing value or expected value functions. The single field `b`
corresponds to the value function with respect to the liquid asset grid.
"""
mutable struct ValueFunctionsOneAsset{T<:AbstractArray} <: ValueFunctions{T}
    b::T
end

"""
    ValueFunctionsTwoAssets

Value function container for the two-asset model.

Fields

  - `b`: value function component for the liquid asset.
  - `k`: value function component for the illiquid asset.
"""
mutable struct ValueFunctionsTwoAssets{T<:AbstractArray} <: ValueFunctions{T}
    b::T
    k::T
end

"""
    ValueFunctionsCompleteMarkets

Placeholder value-function container for complete-markets variants.
"""
mutable struct ValueFunctionsCompleteMarkets{T<:AbstractArray} <: ValueFunctions{T}
    b::T
end

"""
    PolicyFunctionsOneAsset

Policy function container for the one-asset model.

Fields

  - `x_n_star`, `b_n_star`: optimal on-grid policies for composite `x` and liquid `b` in the
    non-adjustment case.
  - `x_tilde_n`, `b_tilde_n`: endogenous-grid policies (interpolants/arrays).
"""
mutable struct PolicyFunctionsOneAsset{T<:AbstractArray} <: PolicyFunctions{T}
    x_n_star::T
    b_n_star::T
    x_tilde_n::T
    b_tilde_n::T
end

"""
    PolicyFunctionsTwoAssets

Policy function container for the two-asset model.

Fields

  - `x_a_star`, `b_a_star`, `k_a_star`: on-grid policies when the agent adjusts the illiquid
    asset.
  - `x_n_star`, `b_n_star`: on-grid policies in the no-adjustment case.
  - `x_tilde_n`, `b_tilde_n`: endogenous-grid policies for the no-adjustment case.
"""
mutable struct PolicyFunctionsTwoAssets{T<:AbstractArray} <: PolicyFunctions{T}
    x_a_star::T
    b_a_star::T
    k_a_star::T
    x_n_star::T
    b_n_star::T
    x_tilde_n::T
    b_tilde_n::T
end

"""
    PolicyFunctionsCompleteMarkets

Placeholder policy-function container for complete-markets variants.
"""
mutable struct PolicyFunctionsCompleteMarkets{T<:AbstractArray} <: PolicyFunctions{T} end

"""
    TransitionMatricesTwoAssets

Transition matrices for two-asset models.

Fields

  - `Γ`: full transition matrix.
  - `Γ_a`: transition matrix for adjusting agents.
  - `Γ_n`: transition matrix for non-adjusting agents.
"""
struct TransitionMatricesTwoAssets{T<:AbstractArray} <: TransitionMatrices{T}
    Γ::T
    Γ_a::T
    Γ_n::T
end

"""
    TransitionMatricesOneAsset

Transition matrices for one-asset models.

Fields

  - `Γ`: transition matrix for the representative stochastic process.
"""
struct TransitionMatricesOneAsset{T<:AbstractArray} <: TransitionMatrices{T}
    Γ::T
end

"""
    TransitionMatricesCompleteMarkets

Placeholder transition container for complete-markets variants.
"""
struct TransitionMatricesCompleteMarkets{T<:AbstractArray} <: TransitionMatrices{T}
    Γ::T
end

"""
    ValueFunctionsOneAssetIndexes

Index container for one-asset value functions (`b` indexes).
"""
struct ValueFunctionsOneAssetIndexes <: ValueFunctionIndexes
    b::AbstractArray{Int,1}
end

"""
    ValueFunctionsTwoAssetsIndexes

Index container for two-asset value functions.

Fields

  - `b`: indexes for liquid-asset value function.
  - `k`: indexes for illiquid-asset marginal value function.
"""
struct ValueFunctionsTwoAssetsIndexes <: ValueFunctionIndexes
    b::AbstractArray{Int,1}
    k::AbstractArray{Int,1}
end

"""
    ValueFunctionsCompleteMarketsIndexes

Index container for complete-markets value functions (placeholder).
"""
struct ValueFunctionsCompleteMarketsIndexes <: ValueFunctionIndexes
    b::AbstractArray{Int,1}
end

"""
    RepAgent

Representative agent distribution values (placeholder).

Field `h` typically stores histogram or aggregated distribution values.
"""
struct RepAgent <: DistributionValues
    h::AbstractArray
end

"""
    RepAgentIndexes

Indexes for `RepAgent` distribution container (placeholder).
"""
struct RepAgentIndexes <: DistributionIndexes
    h::AbstractArray{Int,1}
end

"""
    CDF

Joint cumulative distribution function container.

Field `CDF` stores the joint CDF array over the discretized state space.
"""
mutable struct CDF <: DistributionValues
    CDF::AbstractArray
end

"""
    CDFIndexes

Indexes for `CDF` container.
"""
struct CDFIndexes <: DistributionIndexes
    CDF::Array{Int,1}
end

"""
    CopulaCDFsOneAsset

Copula-based joint CDF components for one-asset models.

Fields

  - `COP`: copula values on grid.
  - `b`, `h`: marginal arrays for corresponding dimensions.
"""
struct CopulaCDFsOneAsset <: CopulaOneAsset
    COP::AbstractArray
    b::AbstractArray
    h::AbstractArray
end

"""
    CopulaPDFsOneAsset

Copula-based joint PDF components for one-asset models.
"""
struct CopulaPDFsOneAsset <: CopulaOneAsset
    COP::AbstractArray
    b::AbstractArray
    h::AbstractArray
end

"""
    CopulaOneAssetIndexes

Indexes for one-asset copula containers.
"""
struct CopulaOneAssetIndexes <: DistributionIndexes
    COP::AbstractArray{Int,1}
    b::AbstractArray{Int,1}
    h::AbstractArray{Int,1}
end

"""
    CopulaPDFsTwoAssets

Copula-based PDF components for two-asset models.
"""
struct CopulaPDFsTwoAssets <: CopulaTwoAssets
    COP::AbstractArray
    b::AbstractArray
    k::AbstractArray
    h::AbstractArray
end

"""
    CopulaCDFsTwoAssets

Copula-based CDF components for two-asset models.
"""
struct CopulaCDFsTwoAssets <: CopulaTwoAssets
    COP::AbstractArray
    b::AbstractArray
    k::AbstractArray
    h::AbstractArray
end

"""
    CopulaTwoAssetsIndexes

Indexes for two-asset copula containers.
"""
struct CopulaTwoAssetsIndexes <: DistributionIndexes
    COP::AbstractArray{Int,1}
    b::AbstractArray{Int,1}
    k::AbstractArray{Int,1}
    h::AbstractArray{Int,1}
end

"""
    TransformationElements

Elements used when transforming and differentiating distributions.

Fields are vectors of matrices/adjoints used in linear transforms and derivatives: `Γ`,
`DC`, `IDC`, `DCD`, and `IDCD`.
"""
struct TransformationElements <: Transformations
    Γ::Vector{Matrix{Float64}}
    DC::Vector{Matrix{Float64}}
    IDC::Vector{Adjoint{Float64,Matrix{Float64}}}
    DCD::Vector{Matrix{Float64}}
    IDCD::Vector{Adjoint{Float64,Matrix{Float64}}}
    pareto_indices::Vector{Vector{Int64}}
end

"""
    SOResults

Second-order solution objects.

Fields

  - `gxx`, `hxx`, `gσσ`, `hσσ`: second-order derivatives or related blocks from
    perturbation/sensitivity calculations.
"""
struct SOResults
    gxx::Any
    hxx::Any
    gσσ::Any
    hσσ::Any
end

"""
    irf

Impulse-response container.

Fields

  - `x`: shock path(s) as a vector or matrix.
  - `y`: response path(s) as a vector or matrix.
"""
struct irf
    x::Union{Vector{Float64},Matrix{Float64}}
    y::Union{Vector{Float64},Matrix{Float64}}
end
