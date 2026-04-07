"""
    run_calibration(moments_function, cal_dict, m_par; solver = "NelderMead")

Calibrate model parameters by minimizing the distance between model-generated moments and
empirical targets.

# Arguments

  - `moments_function`: Function that takes `m_par` and returns a `Dict` of model moments
    (keys matching `cal_dict["target_moments"]`).

  - `cal_dict`: `Dict` with calibration settings. Required keys:

      + `"params_to_calibrate"`: `Vector{Symbol}` of parameter field names to tune.
      + `"target_moments"`: `Dict` of empirical targets keyed by the moment names.
      + `"opt_options"` (optional): optimizer-specific options.
  - `m_par`: model parameter struct (fields referenced by `params_to_calibrate`).
  - `solver`: optimizer selector: `"NelderMead"` (default) or `"BBO"`.

# Returns

  - Updated `m_par` with calibrated parameter values (same struct type as input).

Notes

  - If `moments_function` fails for a candidate parameter vector, the optimizer is given a
    large penalty so it moves away from invalid regions.
"""
function run_calibration(moments_function, cal_dict, m_par; solver = "NelderMead")
    # Unpack all necessary inputs from the calibration dictionary

    println("Starting Calibration")

    # Unpack parameters to calibrate
    params_to_calibrate = cal_dict["params_to_calibrate"]

    # unpack the targets
    target_moments = cal_dict["target_moments"]

    # Unpack optimization options. Otherwise, use default options
    OptOpt =
        haskey(cal_dict, "opt_options") ? cal_dict["opt_options"] :
        generate_default_options(params_to_calibrate, m_par, solver)

    # Get all field names from the model parameter structure.
    field_names = propertynames(m_par)

    # Find indices of parameters to calibrate (matching field names).
    rel_field_ids = [findfirst(x -> x == k, field_names) for k in params_to_calibrate]

    # Extract initial parameter values.
    param_vector = [getfield(m_par, k) for k in params_to_calibrate]

    # Bounds: use user-specified SearchRange if provided; else compute defaults
    bounds = solver == "BBO" ? cal_dict["opt_options"][:SearchRange] : []

    # Objective wrapper: maps optimizer vector -> parameter space -> evaluate moments
    function f_obj(z)
        # To have a smoother optimization landscape
        params = map_z_to_params(z, bounds)

        for (i, idx) in enumerate(rel_field_ids)
            fld_sym = Symbol(field_names[idx])
            m_par = @set m_par.$fld_sym = params[i]
        end

        # The objective function
        return objective_function_for_calibration(moments_function, target_moments, m_par)
    end

    # choice of different optimizers
    opti =
        solver == "NelderMead" ? Optim.optimize(f_obj, param_vector, NelderMead(), OptOpt) :
        solver == "BBO" ? bboptimize(f_obj, param_vector; OptOpt...) :
        error("Unknown solver: $solver, must be 'BBO' or 'NelderMead'")
    result =
        solver == "NelderMead" ? Optim.minimizer(opti) :
        solver == "BBO" ? best_candidate(opti) :
        error("Unknown solver: $solver, must be 'BBO' or 'NelderMead'")

    # Print the fitness value of the best solution
    @printf("Best fitness (squared distance): %.6f\n", f_obj(result))

    # Map back to original parameter space if bounds were used
    result = map_z_to_params(result, bounds)

    for (i, idx) in enumerate(rel_field_ids)
        fld_sym = Symbol(field_names[idx])
        m_par = @set m_par.$fld_sym = result[i]
    end

    # calculate the final model moments
    local final_model_moments
    try
        final_model_moments = moments_function(m_par)
    catch e
        @warn "Final steady state failed; returning penalty moments" exception =
            (e, catch_backtrace())
        final_model_moments = Dict(k => 1e6 for k in keys(target_moments))
    end

    # Print optimized parameters
    for (i, idx) in enumerate(rel_field_ids)
        println("\nOptimized parameter: ", field_names[idx], " = ", result[i])
    end

    print_calibration_results(target_moments, final_model_moments)

    return m_par
end

"""
    objective_function_for_calibration(moments_function, target_moments, m_par)

Compute the (weighted) squared distance between model moments and `target_moments`.

If `moments_function` raises an exception for a candidate `m_par`, a large penalty is
returned so the optimizer avoids invalid regions.
"""
function objective_function_for_calibration(moments_function, target_moments, m_par)
    # creating an empty dict for model moments
    model_moments = Dict()

    # run steady-state solver with updated parameters
    try
        model_moments = moments_function(m_par)
    catch
        model_moments = Dict(k => 1e6 for k in keys(target_moments))
    end

    scales = Dict(
        k => (abs(target_moments[k]) > 0 ? abs(target_moments[k]) : 1.0) for
        k in keys(target_moments)
    )

    # compute the squared distance
    distance = 0.0
    for k in keys(target_moments)
        r = (model_moments[k] - target_moments[k]) / scales[k]
        distance += r^2
    end

    return distance
end

"""
print_calibration_results(target_moments, final_model_moments)

Pretty-print a table comparing target moments and model moments.
"""
function print_calibration_results(target_moments, final_model_moments)
    moment_names = collect(keys(target_moments))

    data = [
        [moment, target_moments[moment], final_model_moments[moment]] for
        moment in moment_names
    ]
    data =
        hcat([row[1] for row in data], [row[2] for row in data], [row[3] for row in data])

    header = ["Moment", "Target", "Model (Optimized)"]

    pretty_table(
        data;
        header,
        formatters = ft_printf("%.3f"),
        alignment = :c,
        header_alignment = :c,
        tf = tf_unicode_rounded,
    )
end

"""
    get_default_searchrange(params_to_calibrate, m_par)

Create reasonable default search ranges for common parameters when `BBO` optimisation is
used.
"""
function get_default_searchrange(params_to_calibrate, m_par)
    param_bounds = Vector{Tuple{Float64,Float64}}(undef, length(params_to_calibrate))
    @printf("Setting default search ranges for parameters:\n")
    for (p, param) in enumerate(params_to_calibrate)
        value = getfield(m_par, param)

        if param == :ξ
            param_bounds[p] = (max(1.0, value * 0.5), value * 2.0)
        elseif param == :γ
            param_bounds[p] = (max(0.1, value * 0.5), min(10.0, value * 2.0))
        elseif param == :β
            param_bounds[p] = (0.90, 0.999)
        elseif param == :λ
            param_bounds[p] = (0.01, 0.5)
        elseif param == :ρ_h
            param_bounds[p] = (0.5, 0.999)
        elseif param == :σ_h
            param_bounds[p] = (0.01, value * 2.0)
        elseif param == :α
            param_bounds[p] = (0.2, 0.5)
        elseif param == :δ_0
            param_bounds[p] = (0.005, 0.1)
        elseif param == :ϕ
            param_bounds[p] = (0.01, value * 2.0)
        elseif param ∈ [:μ, :μw]
            param_bounds[p] = (1.01, 2.0)
        elseif param == :Tlev
            param_bounds[p] = (1.1, 1.9)
        elseif param == :Tprog
            param_bounds[p] = (1.01, 1.5)
        else
            param_bounds[p] = (value * 0.5, value * 2.0)
        end
    end

    return param_bounds
end

"""
    generate_default_options(params_to_calibrate, m_par, solver)

Return a reasonable default options object/dict for the selected `solver`. For `NelderMead`
an `Optim.Options` instance is returned; for `BBO` a keyword-like `Dict` of options is
returned.
"""
function generate_default_options(params_to_calibrate, m_par, solver)
    OptOpt =
        solver == "NelderMead" ?
        Optim.Options(;
            time_limit = length(params_to_calibrate) < 4 ? 3600 : 10800,
            f_reltol = 1e-3,
            store_trace = true,
            show_trace = true,
            show_every = 10,
        ) :
        solver == "BBO" ?
        (
            SearchRange = get_default_searchrange(params_to_calibrate, m_par),
            Method = :adaptive_de_rand_1_bin_radiuslimited,
            MaxTime = length(params_to_calibrate) < 4 ? 3600 : 10800,
            TraceInterval = 30,
            TraceMode = :compact,
            TargetFitness = 1e-3,
        ) : @error("$solver is not in list of solvers: NelderMead, BBO.")

    return OptOpt
end

σ(x) = 1 / (1 + exp(-x))
softplus(x) = log1p(exp(-abs(x))) + max(x, 0)

"""
    map_real_to_param(z, lo, hi)

Map a real scalar `z` to a parameter value constrained to `(lo,hi)`.
"""
function map_real_to_param(z, lo, hi)
    if isfinite(lo) && isfinite(hi)
        return lo + (hi - lo) * σ(z)
    elseif isfinite(lo) && !isfinite(hi)
        return lo + softplus(z)
    elseif !isfinite(lo) && isfinite(hi)
        return hi - softplus(z)
    else
        error("Both bounds are infinite; provide at least one finite bound.")
    end
end

"""
    map_z_to_params(zvec, bounds)

Convert optimizer vector `zvec` into model parameter vector using `bounds`. If `bounds` is
empty the original `zvec` is returned unchanged (suitable for NelderMead).
"""
function map_z_to_params(zvec, bounds)
    if isempty(bounds)
        return zvec
    end
    θ = similar(zvec)
    for i in eachindex(zvec)
        lo, hi = bounds[i]
        if isfinite(lo) && isfinite(hi)
            θ[i] = map_real_to_param(zvec[i], lo, hi)
        elseif isfinite(lo) && !isfinite(hi)
            θ[i] = lo + softplus(zvec[i])
        else
            error("Please provide a finite lower bound for parameter index $i.")
        end
    end
    return θ
end
