"""
    likeli(args...; smoother=false)

Compute the (Gaussian) likelihood of observed data given candidate parameters and priors,
optionally returning Kalman-smoother outputs.

This docstring covers both methods:

  - `likeli(par, Data, Data_missing, H_sel, priors, meas_error, meas_error_std, sr, lr, m_par, e_set; smoother=false)`
  - `likeli(par, sr, lr, er, m_par, e_set; smoother=false)` where `er` bundles data, masks,
    priors, and measurement-error info.

Workflow:

 1. Evaluate the prior; if violated, set `alarm=true` and return a large negative
    likelihood.
 2. Reconstruct model parameters from `par` and build structural (`SCov`) and measurement
    (`MCov`) covariances.
 3. Solve the reduced linear system via `LinearSolution_reduced_system` (aggregate-only
    update).
 4. Form observation matrix `H = H_sel * [I; State2Control]` and compute the likelihood
    using `kalman_filter` (filter only) or `kalman_filter_smoother` (if `smoother=true`).

# Returns

  - If `smoother == false`:

      + `log_like::Float64`: data log-likelihood
      + `prior_like::Float64`: prior log-density
      + `post_like::Float64`: sum of likelihood and prior
      + `alarm::Bool`: true if prior violated or model solution failed
      + `State2Control::AbstractMatrix`: the observation mapping `gx` used to build `H`

  - If `smoother == true`:

      + `smoother_output`: a tuple `(log_lik, xhat_tgt, xhat_tgT, Sigma_tgt, Sigma_tgT, s, m)` from `kalman_filter_smoother`

Notes:

  - Measurement-error treatment is governed by `e_set.me_treatment`:

      + `:fixed` uses `meas_error_std` to set diagonal entries of `MCov` at positions in
        `meas_error`.
      + otherwise, measurement-error stds are read from the tail of `par` in the same order.
"""
function likeli(
    par,
    Data,
    Data_missing,
    H_sel,
    priors,
    meas_error,
    meas_error_std,
    sr,
    lr,
    m_par,
    e_set;
    smoother = false,
)
    return likeli_backend(
        par,
        Data,
        Data_missing,
        H_sel,
        priors,
        meas_error,
        meas_error_std,
        sr,
        lr,
        m_par,
        e_set,
        smoother,
    )
end

function likeli(par, sr, lr, er, m_par, e_set; smoother = false)
    return likeli_backend(
        par,
        er.Data,
        er.Data_missing,
        er.H_sel,
        er.priors,
        er.meas_error,
        er.meas_error_std,
        sr,
        lr,
        m_par,
        e_set,
        smoother,
    )
end

"""
    likeli_backend(par, Data, Data_missing, H_sel, priors, meas_error, meas_error_std, sr, lr, m_par, e_set, smoother)

Implementation of `likeli` shared by both entry-point methods. See `likeli` for high-level
behavior. This function returns either the five-tuple in the filter-only case or the full
smoother output when `smoother=true`.
"""
function likeli_backend(
    par,
    Data,
    Data_missing,
    H_sel,
    priors,
    meas_error,
    meas_error_std,
    sr,
    lr,
    m_par,
    e_set,
    smoother,
)

    # check priors, abort if they are violated
    prior_like::eltype(par), alarm_prior::Bool = prioreval(Tuple(par), Tuple(priors))
    alarm = false
    if alarm_prior
        log_like = -9.e15
        alarm = true
        State2Control = zeros(sr.n_par.ncontrols_r, sr.n_par.nstates_r)
        if e_set.debug_print
            @printf "Parameter try violates PRIOR.\n"
        end
    else
        if e_set.me_treatment != :fixed
            m_start = length(par) - length(meas_error) # find out where in par structural pars end
        else
            m_start = length(par)
        end

        # replace estimated values in m_par by last candidate
        m_par = Flatten.reconstruct(m_par, par[1:m_start])

        # covariance of structural shocks
        SCov = zeros(eltype(par), sr.n_par.nstates_r, sr.n_par.nstates_r)
        for i in e_set.shock_names
            SCov[getfield(sr.indexes_r, i), getfield(sr.indexes_r, i)] =
                (getfield(m_par, Symbol("σ_", i))) .^ 2
        end

        # covariance of measurement errors, assumption: ME ordered after everything else
        m = size(H_sel)[1]
        MCov = diagm(zeros(eltype(par), m)) # no correlated ME allowed for now
        if !isempty(meas_error)
            m_iter = 1
            if e_set.me_treatment != :fixed
                for (k, v) in meas_error # read out position of measurement errors
                    MCov[v, v] = par[m_start + m_iter] .^ 2
                    m_iter += 1
                end
            else
                for (k, v) in meas_error # read out position of measurement errors
                    MCov[v, v] = meas_error_std[m_iter] .^ 2
                    m_iter += 1
                end
            end
        end

        # solve model using candidate parameters
        # BLAS.set_num_threads(1)
        State2Control::Array{eltype(par),2},
        LOM::Array{eltype(par),2},
        alarm_LinearSolution::Bool =
            LinearSolution_reduced_system(sr, m_par, lr.A, lr.B; allow_approx_sol = false)

        # BLAS.set_num_threads(Threads.nthreads())
        if alarm_LinearSolution # abort if model doesn't solve
            log_like = -9.e15
            alarm = true
            if e_set.debug_print
                @printf "Parameter try leads to inexistent or unstable equilibrium.\n"
            end
        else
            MX = [I; State2Control]
            H = H_sel * MX
            if smoother == false
                log_like = kalman_filter(H, LOM, Data, Data_missing, SCov, MCov, e_set)
                # log_like = kalman_filter_herbst(Data, LOM, SCov, H, MCov, 0, e_set)
            else
                smoother_output =
                    kalman_filter_smoother(H, LOM, Data, .!Data_missing, SCov, MCov, e_set)
                log_like = smoother_output[1]
            end
        end
    end
    post_like = log_like + prior_like

    if smoother == false
        return log_like, prior_like, post_like, alarm, State2Control
    else
        return smoother_output
    end
end

"""
    prioreval(par, priors)

Evaluate the joint prior density at parameters `par` using Distributions.jl.

# Arguments

  - `par::Tuple` or vector: parameter values
  - `priors::Tuple` or vector: matching prior distributions

# Returns

  - `log_priorval::Float64`: sum of log-densities if all `par[i] ∈ support(priors[i])`,
    otherwise `-9e15`
  - `alarm::Bool`: `true` if any parameter lies outside its prior support
"""
function prioreval(par, priors)
    if all(insupport.(priors, par))
        alarm = false
        log_priorval = sum(logpdf.(priors, par))
    else
        alarm = true
        log_priorval = -9.e15
    end

    return log_priorval, alarm
end
