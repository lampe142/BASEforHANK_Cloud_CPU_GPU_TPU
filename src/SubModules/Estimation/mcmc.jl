"""
    rwmh(xhat, Σ, sr, lr, er, m_par, e_set)

Random-Walk Metropolis–Hastings sampler for the posterior of the model parameters.

This routine draws `e_set.ndraws + e_set.burnin` proposals using a multivariate normal
random walk with proposal covariance `Σ`, evaluates the (log) posterior using either
`likeli` (default) or `irfmatch` (if `e_set.irf_matching == true`), and accepts/rejects via
the usual MH rule.

# Arguments

  - `xhat::Vector{Float64}`: initial parameter vector (at least approximately feasible)
  - `Σ::Symmetric{Float64,Array{Float64,2}}`: proposal covariance matrix for the random walk
  - `sr, lr, er, m_par, e_set`: model structures and estimation settings; see package docs

# Returns

  - `draws::Matrix{Float64}`: `(e_set.ndraws + e_set.burnin) × length(xhat)` matrix of
    draws, each row a parameter vector
  - `posterior::Vector{Float64}`: log posterior for each stored draw (likelihood + prior)
  - `accept_rate::Float64`: overall acceptance rate

Notes:

  - When `e_set.irf_matching == false`, the posterior is computed with `likeli`.
  - When `e_set.irf_matching == true`, IRF targets are read from a CSV according to
    `e_set.irf_matching_dict`, and the posterior is computed with `irfmatch` using
    inverse-variance weights. The function periodically prints progress (draw count,
    acceptance rate, posterior) and the current parameters.
"""
function rwmh(
    xhat::Vector{Float64},
    Σ::Symmetric{Float64,Array{Float64,2}},
    sr,
    lr,
    er,
    m_par,
    e_set,
)
    NormDist = MvNormal(zeros(length(xhat)), Σ)
    accept = 0
    accept_rate = 0.0
    draws = Matrix{Float64}(undef, e_set.ndraws + e_set.burnin, length(xhat))
    posterior = Vector{Float64}(undef, e_set.ndraws + e_set.burnin)
    draws[1, :] = xhat

    if !e_set.irf_matching
        old_posterior, alarm = likeli(xhat, sr, lr, er, m_par, e_set)[3:4]
        posterior[1] = copy(old_posterior)
        proposal_draws = e_set.mhscale .* rand(NormDist, e_set.ndraws + e_set.burnin)
        for i = 2:(e_set.ndraws + e_set.burnin)
            xhatstar = draws[i - 1, :] .+ proposal_draws[:, i]
            new_posterior, alarm = likeli(xhatstar, sr, lr, er, m_par, e_set)[3:4]

            accprob = min(exp(new_posterior - old_posterior), 1.0)
            if alarm == false && rand() .<= accprob
                draws[i, :] = xhatstar
                posterior[i] = copy(old_posterior)
                old_posterior = new_posterior
                accept += 1
            else
                draws[i, :] = draws[i - 1, :]
                posterior[i] = posterior[i - 1]
            end
            if mod(i, 200) == 0 || i == e_set.ndraws + e_set.burnin
                @printf("-----------------------\n")
                @printf "\n"
                pretty_table(
                    [
                        "Number of draws" i
                        "Acceptance Rate" @sprintf("%.4f", accept / i)
                        "Posterior Likelihood" @sprintf("%.4f", old_posterior)
                    ];
                    header = ["Metric", "Value"],
                    title = "Simulation status",
                )
                @printf("Parameters\n")
                @printf("%s\n", string(draws[i, :]))
                @printf("-----------------------\n")
                accept_rate = accept / i
            end
        end

        return draws, posterior, accept_rate

    elseif e_set.irf_matching
        irf_horizon = e_set.irf_matching_dict["irf_horizon"]

        # Load irf targets
        key_to_use =
            haskey(e_set.irf_matching_dict, "irfs_to_target") ? "irfs_to_target" :
            "irfs_to_plot"

        Data_temp =
            DataFrame(CSV.File(e_set.irf_matching_dict[key_to_use]; missingstring = "NaN"))
        shock_names = Symbol.(unique(Data_temp[:, :shock]))
        shocks_selected = intersect(shock_names, e_set.shock_names)
        select_variables =
            intersect(Symbol.(propertynames(Data_temp)), e_set.observed_vars_input)

        IRFtargets = Array{Float64}(
            undef,
            irf_horizon,
            length(select_variables),
            length(shocks_selected),
        )
        IRF_sderr = Array{Float64}(
            undef,
            irf_horizon,
            length(select_variables),
            length(shocks_selected),
        )

        count_shock = 0
        for i in shocks_selected
            count_shock += 1
            count_outcm = 0
            for j in select_variables
                count_outcm += 1
                IRFtargets[:, count_outcm, count_shock] = Data_temp[
                    (Data_temp[
                        :,
                        :pointdum,
                    ] .== 1) .& (Symbol.(Data_temp[:, :shock]) .== i),
                    j,
                ]
                IRF_sderr[:, count_outcm, count_shock] = Data_temp[
                    (Data_temp[
                        :,
                        :pointdum,
                    ] .== 0) .& (Symbol.(Data_temp[:, :shock]) .== i),
                    j,
                ]
            end
        end
        var_to_scale_by = e_set.irf_matching_dict["scale_responses_by"]
        scale_term =
            var_to_scale_by === nothing ? 1 : maximum(Data_temp[:, var_to_scale_by])

        IRFtargets = IRFtargets ./ scale_term
        IRF_sderr = IRF_sderr ./ scale_term

        # To weight the IRF targets
        # Weights are the inverse of the standard deviation of the IRF targets
        weights = 1.0 ./ (IRF_sderr .^ 2)
        iter = 1
        indexes_sel_vars = []
        isstate = zeros(Bool, length(select_variables))
        for i in select_variables
            if i in Symbol.(sr.state_names)
                isstate[iter] = true
            end
            iter += 1
            append!(indexes_sel_vars, getfield(sr.indexes_r, i))
        end
        priors = collect(metaflatten(m_par, prior)) # model parameters

        old_posterior, alarm = irfmatch(
            xhat,
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
        )[3:4]

        posterior[1] = copy(old_posterior)
        proposal_draws = e_set.mhscale .* rand(NormDist, e_set.ndraws + e_set.burnin)
        for i = 2:(e_set.ndraws + e_set.burnin)
            xhatstar = draws[i - 1, :] .+ proposal_draws[:, i]
            new_posterior, alarm = irfmatch(
                xhatstar,
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
            )[3:4]

            accprob = min(exp(new_posterior - old_posterior), 1.0)
            if alarm == false && rand() .<= accprob
                draws[i, :] = xhatstar
                posterior[i] = new_posterior
                old_posterior = new_posterior
                accept += 1
            else
                draws[i, :] = draws[i - 1, :]
                posterior[i] = posterior[i - 1]
            end
            if mod(i, 200) == 0 || i == e_set.ndraws + e_set.burnin
                unmute_printf("-----------------------\n\n")

                pretty_table(
                    Tools.ORIG_STDOUT,   # <—— force it to print to the saved stdout
                    [
                        "Number of draws" i
                        "Acceptance Rate" @sprintf("%.4f", accept / i)
                        "Posterior Likelihood" @sprintf("%.4f", old_posterior)
                    ];
                    header = ["Metric", "Value"],
                    title = "Simulation status",
                )

                unmute_printf("Parameters\n")
                unmute_printf("%s\n", string(draws[i, :]))
                unmute_printf("-----------------------\n")
                accept_rate = accept / i
            end
        end

        return draws, posterior, accept_rate
    end
end

"""
    multi_chain_init(xhat, Σ, sr, lr, er, m_par, e_set)

Construct an overdispersed starting value for a Metropolis–Hastings chain.

Draws proposals from `MvNormal(0, Σ)` scaled by `2 * e_set.mhscale` around `xhat` and keeps
the first candidate that yields a feasible model (no alarm from `likeli`). Tries up to 100
times.

# Arguments

  - `xhat::Vector{Float64}`: baseline parameter vector
  - `Σ::Symmetric{Float64,Array{Float64,2}}`: covariance used to generate overdispersed
    proposals
  - `sr, lr, er, m_par, e_set`: model structures and estimation settings

# Returns

  - `init_draw::Vector{Float64}`: overdispersed, feasible starting parameter vector
  - `init_success::Bool`: whether a feasible start was found within 100 attempts
"""
function multi_chain_init(
    xhat::Vector{Float64},
    Σ::Symmetric{Float64,Array{Float64,2}},
    sr,
    lr,
    er,
    m_par,
    e_set,
)
    init_scale = 2 * e_set.mhscale # overdispersed initial values
    NormDist = MvNormal(zeros(length(xhat)), Σ)
    init_draw = Vector{Float64}(undef, length(xhat))
    init_success = false
    init_iter = 1
    while init_success == false && init_iter <= 100
        init_draw .= init_scale^2.0 .* rand(NormDist) .+ xhat

        alarm = likeli(init_draw, sr, lr, er, m_par, e_set)[4]
        if alarm == false
            init_success = true
        else
            init_iter += 1
        end
    end

    return init_draw, init_success
end

"""
    marginal_likeli(draws, posterior)

Estimate the log marginal likelihood using the Modified Harmonic Mean estimator of Geweke
(1998), averaging over a grid of truncation levels.

Given MCMC draws `θ_i` and their log posteriors `posterior[i]`, the estimator computes
Gaussian kernel densities within a χ²-ball defined by `τ ∈ {0.1, …, 0.9}` and averages the
implied marginal-likelihood estimates for stability.

# Arguments

  - `draws::AbstractMatrix{<:Real}`: `ndraws × npars` matrix of parameter draws
  - `posterior::AbstractVector{<:Real}`: log posterior evaluations for the draws

# Returns

  - `marg_likeli::Float64`: estimated log marginal likelihood

Caution: As with all harmonic-mean style estimators, results can be sensitive to tail
behavior and mixing. Use with diagnostics and, if possible, alternative estimators.
"""
function marginal_likeli(draws, posterior)
    ndraws, npars = size(draws)
    posterior_mode = maximum(posterior)
    d = Chisq(npars)
    θ_hat = mean(draws; dims = 1)[:]
    V_hat = cov(draws)
    inv_V_hat = inv(V_hat)

    marg_likeli_save = zeros(9)
    τ_iter = 1
    for τ = 0.1:0.1:0.9
        thresh = quantile(d, τ)
        const_terms = -0.5 * npars * log(2 * pi) - 0.5 * logdet(V_hat) - log(τ)

        tmp = 0.0
        for i = 1:ndraws
            θ_dist = (draws[i, :] .- θ_hat)' * inv_V_hat * (draws[i, :] .- θ_hat)
            if θ_dist <= thresh
                log_f_θ = const_terms - 0.5 * θ_dist
                tmp += exp(log_f_θ - posterior[i] + posterior_mode)
            end
        end
        marg_likeli_save[τ_iter] = posterior_mode - log(tmp / ndraws)
        τ_iter += 1
    end
    marg_likeli = mean(marg_likeli_save)

    return marg_likeli
end
