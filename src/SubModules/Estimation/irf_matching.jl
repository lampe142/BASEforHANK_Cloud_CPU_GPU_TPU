"""
    irfmatch(par, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)

User-facing entry point for Bayesian IRF matching in the style of Christiano et al.

This wrapper remaps the candidate parameter vector `par` to the support of the specified
`priors` and delegates to `irfmatch_backend` to evaluate the IRF distance and prior
contribution.

# Arguments

  - `par::AbstractVector`: candidate parameters (unconstrained space allowed).
  - `IRFtargets::AbstractArray`: target IRFs with shape `(T, n_vars, n_shocks)`.
  - `weights::AbstractArray`: weights (same shape as `IRFtargets`) applied to squared
    errors.
  - `shocks_selected::Vector{Symbol}`: names of shocks to perturb.
  - `isstate::BitVector`: mask of selected variables that are states (end-of-period
    reporting).
  - `indexes_sel_vars::AbstractVector{Int}`: indexes of selected variables in `[k; d]`.
  - `priors::AbstractVector`: prior distributions for parameters (Distributions.jl).
  - `sr`, `lr`: steady and linear results (supply indexes and linear solution).
  - `m_par`: model parameters struct (baseline).
  - `e_set`: estimation settings (e.g., horizon under `irf_matching_dict`).

# Returns

  - `llike_irf::Float64`: negative IRF distance (higher is better).
  - `prior_like::Float64`: prior log-density for `par`.
  - `post_like::Float64`: objective combining IRF fit and prior (scaled prior).
  - `alarm::Bool`: true if prior violated or model solution failed.
"""
function irfmatch(
    par,
    IRFtargets,
    weights,
    shocks_selected,
    isstate,
    indexes_sel_vars,
    priors,
    sr,
    lr,
    m_par,
    e_set,
)
    par = remap_params!(par, priors)

    return irfmatch_backend(
        par,
        IRFtargets,
        weights,
        shocks_selected,
        isstate,
        indexes_sel_vars,
        priors,
        sr,
        lr,
        m_par,
        e_set,
    )
end

"""
     irfmatch_backend(par, IRFtargets, weights, shocks_selected, isstate, indexes_sel_vars, priors, sr, lr, m_par, e_set)

Core IRF-matching routine. Evaluates the weighted squared distance between model-implied
IRFs and `IRFtargets` and combines it with the prior log-density.

Workflow:

 1. Evaluate prior via `prioreval`; if violated, set a large distance and raise `alarm`.
 2. Reconstruct `m_par` from `par` and solve the reduced linear system using
    `LinearSolution_reduced_system` (aggregate-only update).
 3. Compute IRFs for `shocks_selected` and `indexes_sel_vars` for the horizon
    `e_set.irf_matching_dict["irf_horizon"]`.
 4. IRF distance is 0.5 × sum of weighted squared deviations; objective is distance plus
    prior.

# Returns

  - `llike_irf::Float64`: negative distance (−IRFdist).
  - `prior_like::Float64`: prior log-density.
  - `post_like::Float64`: `−IRFdist + prior_scale * prior_like`.
  - `alarm::Bool`: true if prior violated or solution failed.
"""
function irfmatch_backend(
    par,
    IRFtargets,
    weights,
    shocks_selected,
    isstate,
    indexes_sel_vars,
    priors,
    sr,
    lr,
    m_par,
    e_set,
)

    # check priors, abort if they are violated
    prior_like::eltype(par), alarm_prior::Bool = prioreval(Tuple(par), Tuple(priors))
    alarm = false
    if alarm_prior
        IRFdist = 9.e15
        alarm = true
        State2Control = zeros(sr.n_par.ncontrols_r, sr.n_par.nstates_r)
    else
        # replace estimated values in m_par by last candidate
        m_par = Flatten.reconstruct(m_par, par)

        # solve model using candidate parameters
        local State2Control, LOMstate, alarm_sgu
        @silent begin
            State2Control, LOMstate, alarm_sgu = LinearSolution_reduced_system(
                sr,
                m_par,
                lr.A,
                lr.B;
                allow_approx_sol = true,
            )
        end
        if alarm_sgu # abort if model doesn't solve
            IRFdist = 9.e15
            alarm = true
        else
            irf_horizon = e_set.irf_matching_dict["irf_horizon"]
            IRFs = compute_irfs_for_matching(
                sr,
                State2Control,
                LOMstate,
                m_par,
                shocks_selected,
                indexes_sel_vars,
                isstate,
                irf_horizon,
            )
            IRFdist = (sum((IRFs[:] .- IRFtargets[:]) .^ 2 .* weights[:])) ./ 2
        end
    end

    prior_scale = 1.0 # could be adjusted to tune the relative weight of prior i.e., 0.0 is MLE

    return -IRFdist, prior_like, -IRFdist .+ prior_like * prior_scale, alarm
end

"""
    compute_irfs_for_matching(sr, State2Control, LOMstate, m_par, shocks_selected, indexes_sel_vars, isstate, irf_horizon)

Compute model-implied impulse responses for selected variables and shocks using the linear
solution `(State2Control, LOMstate)`.

Conventions:

  - For state variables (`isstate = true`), IRFs are reported at end-of-period (t = 1…H).
  - For control variables, IRFs are reported at the beginning-of-period (aligned with states
    one period earlier).

# Arguments

  - `sr`: steady results (provides `indexes_r` for shock positions).
  - `State2Control::AbstractMatrix`: `gx` mapping states to controls.
  - `LOMstate::AbstractMatrix`: `hx` mapping states to next states.
  - `m_par`: model parameters (provides shock scales `σ_<shock>`).
  - `shocks_selected::Vector{Symbol}`: shocks to apply (one at a time).
  - `indexes_sel_vars::AbstractVector{Int}`: indexes of variables for which to record IRFs.
  - `isstate::BitVector`: mask of `indexes_sel_vars` that are states.
  - `irf_horizon::Integer`: number of periods to report.

# Returns

  - `IRFsout::Array{Float64,3}` of size `(irf_horizon, n_vars, n_shocks)`.
"""
function compute_irfs_for_matching(
    sr,
    State2Control,
    LOMstate,
    m_par,
    shocks_selected,
    indexes_sel_vars,
    isstate,
    irf_horizon,
)
    n_vars = length(indexes_sel_vars)
    n_shocks = length(shocks_selected)

    IRFs = Array{Float64}(undef, irf_horizon + 1, n_vars, n_shocks)
    IRFsout = Array{Float64}(undef, irf_horizon, n_vars, n_shocks)

    for (i, s) in enumerate(shocks_selected)
        x = zeros(size(LOMstate, 1))
        x[getfield(sr.indexes_r, s)] = getfield(m_par, Symbol("σ_", s))

        MX = [I; State2Control]
        for t = 1:(irf_horizon + 1)
            IRFs[t, :, i] = (MX[indexes_sel_vars, :] * x)'
            x[:] = LOMstate * x
        end
    end
    IRFsout[:, isstate, :] .= IRFs[2:end, isstate, :] # IRFs for state variables represent end-of-period values
    IRFsout[:, .~isstate, :] .= IRFs[1:(end - 1), .~isstate, :] # IRFs for state variables represent end-of-period values

    return IRFsout
end

"""
    softplus(x)

Numerically stable mapping to the positive reals: `softplus(x) = log(1 + exp(x))`. Used to
map unconstrained parameters to the support of Gamma/InverseGamma priors.
"""
softplus(x) = log(1 + exp(x))

"""
    remap_params!(θ, priors; ϵ = 1e-9)

Project a parameter vector `θ` onto the support of the provided `priors` in-place.

Mappings when `θ[i]` lies outside `support(priors[i])`:

  - `Gamma`/`InverseGamma`: `θ[i] = softplus(θ[i]) + ϵ` (ensures > 0).
  - `Beta`: `θ[i] = ϵ + (1 − 2ϵ) / (1 + exp(−θ[i]))` (maps to (ϵ, 1 − ϵ)).
  - `Normal`: unbounded; left unchanged.

# Arguments

  - `θ::AbstractVector`: parameter vector to be remapped (modified in-place).
  - `priors::AbstractVector`: vector of distributions compatible with Distributions.jl.
  - `ϵ::Real`: small positive slack to avoid boundary issues (default `1e-9`).

# Returns

  - The modified vector `θ` (returned for convenience).
"""
function remap_params!(θ::AbstractVector, priors::AbstractVector; ϵ = 1e-9)
    @assert length(θ) == length(priors)
    for i in eachindex(θ)
        d = priors[i]
        if θ[i] ∉ Distributions.support(d)
            if d isa Gamma || d isa InverseGamma
                θ[i] = softplus(θ[i]) + ϵ                     # > 0

            elseif d isa Beta
                θ[i] = ϵ + (1 - 2ϵ) / (1 + exp(-θ[i]))        # ∈ (ϵ, 1-ϵ)

            elseif d isa Normal
                continue  # unbounded
            end
        end
    end

    return θ
end
