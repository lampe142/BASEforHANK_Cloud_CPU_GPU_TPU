"""
    measurement_error(Data, observed_vars, e_set)

Construct the measurement-error specification from user settings and data.

Maps each observable listed in `e_set.meas_error_input` to its column index in
`observed_vars`, and builds corresponding priors and standard-deviation caps according to
`e_set.me_treatment`.

# Arguments

  - `Data::AbstractMatrix`: observed data matrix of size `nobs × nvar`
  - `observed_vars::AbstractVector{Symbol}`: names of observables, length `nvar`
  - `e_set::EstimationSettings`: estimation settings controlling measurement errors

# Returns

  - `meas_error::OrderedDict{Symbol,Int}`: mapping from observable names to their column
    index
  - `meas_error_prior::Vector{<:Distribution}`: prior distributions for measurement-error
    stds
  - `meas_error_std::Vector{Float64}`: per-observable std caps (used when `me_treatment ∈ (:bounded, :fixed)`) or defaults

Treatment modes (`e_set.me_treatment`):

  - `:unbounded`: uses `e_set.meas_error_distr` (e.g., inverse-gamma) as priors and sets
    `meas_error_std .= e_set.me_std_cutoff` (a neutral scaling anchor).
  - `:bounded` or `:fixed`: sets a data-driven upper bound for each selected observable as
    `UB_j = e_set.me_std_cutoff * std(skipmissing(Data[:, j]))` and assigns `Uniform(0, UB_j)` priors. For `:fixed`, these caps are later used directly as fixed stds.

If `e_set.meas_error_input` is empty, returns an empty `OrderedDict()`, an empty vector of
priors, and an empty `Float64[]` for stds.
"""
function measurement_error(Data, observed_vars, e_set)
    if !isempty(e_set.meas_error_input)

        # find correct positions for measurement error
        meas_error_index = Vector{Int}(undef, length(e_set.meas_error_input))
        for i in eachindex(e_set.meas_error_input)
            meas_error_index[i] =
                findall(x -> x == e_set.meas_error_input[i], observed_vars)[1]
        end
        meas_error = OrderedDict(zip(e_set.meas_error_input, meas_error_index))

        # create measurement error according to selected treatment
        if e_set.me_treatment == :unbounded
            # inverse gamma prior
            meas_error_prior = e_set.meas_error_distr
            meas_error_std = e_set.me_std_cutoff * ones(length(e_set.meas_error_input))
        elseif e_set.me_treatment == :bounded || e_set.me_treatment == :fixed
            # data dependent hard upper bound on measurement error standard deviation
            meas_error_prior =
                Array{Uniform{Float64}}(undef, length(e_set.meas_error_input))
            meas_error_std = Vector{Float64}(undef, length(e_set.meas_error_input))
            m_iter = 1
            for (k, v) in meas_error # read out position of measurement errors
                meas_error_std[m_iter] = e_set.me_std_cutoff * std(skipmissing(Data[:, v]))
                meas_error_prior[m_iter] = Uniform(0.0, meas_error_std[m_iter])
                m_iter += 1
            end
        else
            error("ME treatment not implemented")
        end
    else
        # in case of no measurement error
        meas_error = OrderedDict{Symbol,Int}()
        meas_error_prior = repeat([InverseGamma(ig_pars(0.0005, 0.001^2)...)], 0)
        meas_error_std = Vector{Float64}(undef, 0)
    end

    return meas_error, meas_error_prior, meas_error_std
end
