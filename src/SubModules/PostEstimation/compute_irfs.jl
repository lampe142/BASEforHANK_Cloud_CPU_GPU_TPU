"""
    compute_irfs(
      exovars,
      gx,
      hx,
      XSS,
      ids;
      T = 1000,
      init_val = fill(0.01, length(exovars)),
      verbose = true,
      irf_interval_options = nothing,
      distribution = false,
      transform_elements = nothing,
      comp_ids = nothing,
      n_par = nothing,
      m_par = nothing,
    )

Computes impulse response functions (IRFs) for a given set of shocks to exogenous variables.

# Arguments

  - `exovars::Vector{Int64}`: Vector of positional indices of exogenous variables to which
    shocks are applied.
  - `gx::Matrix{Float64}`: Control matrix (states to controls equations)
  - `hx::Matrix{Float64}`: Transition matrix for states (state transition equations)
  - `XSS::Vector{Float64}`: Steady state values of the model variables.
  - `ids`: Indexes of the model variables.

# Keyword Arguments

  - `T::Int64`: Number of periods for which to compute IRFs.
  - `init_val::Vector{Float64}`: Initial value of the shock to each exogenous variable,
    defaults to 0.01 for all shocks.
  - `verbose::Bool`: Print progress to console.
  - `irf_interval_options`: If provided, computes confidence intervals for IRFs based on
    parameter draws. Should be a dictionary with keys `"draws"` (matrix of parameter draws)
    and `"e_set"` (estimation settings struct).
  - `distribution::Bool`: Compute distributional IRFs, defaults to false.
  - `comp_ids`: Compression indices for the distribution, as created by
    `prepare_linearization`. Needed if `distribution` is true. Defaults to nothing.
  - `n_par`: Model parameters, needed for distributional IRFs. Defaults to nothing.
  - `m_par`: Model parameters, needed for distributional IRFs. Defaults to nothing.
  - `transform_elements`: Transformation elements for the distributional IRFs. Defaults to nothing.

# Returns

  - `IRFs`, `IRFs_lvl`: 3D array of (level) IRFs for each exogenous variable, with
    dimensions:

      + 1: States and controls
      + 2: Time periods
      + 3: Exogenous variables

  - `IRFs_order`: Names of the exogenous variables and their indices in the IRFs array.
  - `IRFs_dist`: Dictionary containing the full distributional IRFs, if requested.
"""
function compute_irfs(
    exovars::Vector{Int64},
    gx::Matrix{Float64},
    hx::Matrix{Float64},
    XSS::Vector{Float64},
    ids;
    T::Int64 = 1000,
    init_val::Vector{Float64} = fill(0.01, length(exovars)),
    verbose::Bool = true,
    irf_interval_options = nothing,
    distribution::Bool = false,
    transform_elements = nothing,
    comp_ids = nothing,
    n_par = nothing,
    m_par = nothing,
)

    # If distributional IRFs are requested, check if all required arguments are provided
    if distribution && (isnothing(comp_ids) || isnothing(n_par))
        throw(
            ArgumentError(
                "Missing arguments: `comp_ids` and `n_par` are required for distributional IRFs.",
            ),
        )
    end

    # Compute the number of states and controls from gx and hx
    ncontrols = size(gx, 1)
    nstates = size(hx, 1)
    nvars = nstates + ncontrols

    # Initialize IRFs for all selected exogenous variables
    IRFs = zeros(nvars, T, length(exovars))
    IRFs_lvl = similar(IRFs)
    if distribution
        IRFs_dist = initialize_distributional_dict(n_par, T, length(exovars))
    end

    # Generate irf matching objects if provided -- nothing otherwise
    if !isnothing(irf_interval_options)
        draws_after_burnin =
            irf_interval_options["draws"][(1 + irf_interval_options["e_set"].burnin):end, :]
        IRF_lb = similar(IRFs)
        IRF_ub = similar(IRFs)
        IRF_lvl_lb = similar(IRFs)
        IRF_lvl_ub = similar(IRFs)
    end

    # Store the shock names
    IRFs_order = [find_field_with_value(ids, exovars[i], false) for i = 1:length(exovars)]

    # Compute IRFs for each exogenous variable
    for (i, exovar) in enumerate(exovars)
        if verbose
            @printf "Computing IRFs for %s with index %d and initial condition of %f.\n" IRFs_order[i] exovar init_val[i]
        end
        if distribution
            IRFs[:, :, i], IRFs_lvl[:, :, i], dist_results = compute_irfs_inner(
                exovars[i],
                gx,
                hx,
                XSS,
                ids,
                T,
                nstates,
                ncontrols,
                init_val[i],
                distribution,
                transform_elements,
                comp_ids,
                n_par,
                m_par,
            )

            # Fill in here
            for (name, data) in pairs(dist_results)
                colon_tuple = (ntuple(_ -> Colon(), ndims(data))..., i)
                IRFs_dist[name][colon_tuple...] = data
            end

        else
            IRFs[:, :, i], IRFs_lvl[:, :, i] = compute_irfs_inner(
                exovars[i],
                gx,
                hx,
                XSS,
                ids,
                T,
                nstates,
                ncontrols,
                init_val[i],
                distribution,
                transform_elements,
                comp_ids,
                n_par,
                m_par,
            )

            # Computing IRFs for all parameter draws (for IRF matching)
            if !isnothing(irf_interval_options)
                IRF_lb[:, :, i], IRF_ub[:, :, i], IRF_lvl_lb[:, :, i], IRF_lvl_ub[:, :, i] =
                    compute_irf_confidence_intervals(
                        draws_after_burnin,
                        exovars[i],
                        init_val[i],
                        nvars,
                        T,
                        irf_interval_options,
                    )
            end
        end
    end

    if distribution
        IRFs_dist = Dict(
            key => convert(Array{Float64}, value) for
            (key, value) in IRFs_dist if value isa AbstractArray
        )
        return IRFs, IRFs_lvl, IRFs_order, IRFs_dist
    else
        if !isnothing(irf_interval_options)
            return IRFs, IRFs_lvl, IRF_lb, IRF_ub, IRF_lvl_lb, IRF_lvl_ub, IRFs_order
        else
            return IRFs, IRFs_lvl, IRFs_order
        end
    end
end

"""
    initialize_distributional_dict(n_par, T, n_shocks)

Create and return a dictionary of zero-initialized arrays intended for storing
distributional impulse response functions (IRFs) and related marginal / joint objects.

Parameters

  - `n_par` : Model parameters struct, needed for distributional IRFs. Contains integer
    fields `nb`, `nk`, `nh`

      + `nb` = number of grid points for state `b`
      + `nk` = number of grid points for state `k`
      + `nh` = number of grid points for state `h`

  - `T` : Integer

      + Time horizon (number of periods) for the IRFs
  - `n_shocks` : Integer

      + Number of shocks for which IRFs are computed

Returns

  - `Dict{String,Any}` mapping descriptive names to zero-initialized arrays (Float64) with
    shapes depending on the object:

      + Base 3D objects (named `"Wb"`, `"Wk"`, `"PDF"`): (nb, nk, nh, T, n_shocks)
      + 1D marginal objects (constructed as `"<base>_b"`, `"<base>_k"`, `"<base>_h"`):    #
        Define the base dimensions (3D objects) (nb, T, n_shocks), (nk, T, n_shocks), (nh,
        T, n_shocks)
      + 2D joint objects (constructed as `"<base>_bk"`, `"<base>_bh"`, `"<base>_kh"`): (nb,
        nk, T, n_shocks), (nb, nh, T, n_shocks), (nk, nh, T, n_shocks)    # Define the 1D
        objects (e.g., PDF_b)
"""
function initialize_distributional_dict(n_par, T, n_shocks)
    # Define the base dimensions (3D objects)
    dims_3d = (n_par.nb, n_par.nk, n_par.nh, T, n_shocks)

    # Define the 1D objects (e.g., PDF_b)
    dims_1d = Dict(
        "b" => (n_par.nb, T, n_shocks),
        "k" => (n_par.nk, T, n_shocks),
        "h" => (n_par.nh, T, n_shocks),
    )

    # Define the 2D objects (e.g., PDF_bk)
    dims_2d = Dict(
        "bk" => (n_par.nb, n_par.nk, T, n_shocks),
        "bh" => (n_par.nb, n_par.nh, T, n_shocks),
        "kh" => (n_par.nk, n_par.nh, T, n_shocks),
    )

    IRFs_dist = Dict{String,Any}()

    # Base 3D objects
    for name in ["Wb", "Wk", "PDF"]
        IRFs_dist[name] = zeros(dims_3d...)
    end

    # Marginal/Joint objects
    dist_to_dims = [
        ("PDF", dims_1d),
        ("Wb", dims_1d),
        ("Wk", dims_1d),
        ("PDF", dims_2d),
        ("Wb", dims_2d),
        ("Wk", dims_2d),
    ]
    for (key, dims_map) in dist_to_dims
        for (suffix, dims) in dims_map
            name = string(key, "_", suffix) # "PDF_b", "Wk_kh", etc.
            IRFs_dist[name] = zeros(dims...)
        end
    end

    return IRFs_dist
end

"""
    compute_irfs_inner(
        exovars::Int64,
        gx::Matrix{Float64},
        hx::Matrix{Float64},
        XSS::Vector{Float64},
        ids,
        T::Int64,
        nstates::Int64,
        ncontrols::Int64,
        init_val::Float64,
        distribution::Bool,
        transform_elements,
        comp_ids,
        n_par,
        m_par,
    )

Simulates impulse response functions (IRFs) for a single shock to a linear state-space
system and reconstructs the levels for a subset of aggregate variables.

# Arguments

  - `exovar::Int64`: Positional index of the exogenous state variable receiving the initial
    shock.
  - `gx::Matrix{Float64}`: Control matrix mapping states to controls (size: ncontrols ×
    nstates).
  - `hx::Matrix{Float64}`: State transition matrix (size: nstates × nstates).
  - `XSS::Vector{Float64}`: Vector of steady-state values used to reconstruct levels.
  - `ids`: A struct or other mapping object providing indices for model variables.

# Keyword Arguments

  - `T::Int64`: Number of periods to simulate the IRF.
  - `nstates::Int64`: Number of state variables.
  - `ncontrols::Int64`: Number of control variables.
  - `init_val::Float64`: Initial value of the shock applied to the `exovar` state.
  - `distribution::Bool`: If true, computes additional distributional IRFs.
  - `transform_elements`: Transformation elements (DCT, IDCT, Γ) needed for distributional
    IRFs.
  - `comp_ids`: Compression indices passed to the distributional IRF routine.
  - `n_par`: Model parameters (e.g., grid sizes) passed to the distributional IRF routine.
  - `m_par`: Model parameters struct passed to the distributional IRF routine.

# Returns

  - If `distribution == false`, returns a tuple `(original, level)`:

      + `original::Matrix{Float64}`: A matrix of IRFs in deviations from the steady state,
        with dimensions `(nstates + ncontrols) × T`.
      + `level::Matrix{Float64}`: A matrix of the same dimensions containing IRFs in levels
        for aggregate variables and `NaN` otherwise.

  - If `distribution == true`, returns a tuple `(original, level, dist_results)`:

      + `original`, `level`: As above.
      + `dist_results`: The dictionary of distributional IRFs returned by
        `compute_irfs_inner_distribution`.

# Notes

  - This function performs a deterministic simulation of the linear system after the initial
    shock.
  - It relies on a globally defined `aggr_names` vector to identify which variables to
    reconstruct in levels.
"""
function compute_irfs_inner(
    exovar::Int64,
    gx::Matrix{Float64},
    hx::Matrix{Float64},
    XSS::Vector{Float64},
    ids,
    T::Int64,
    nstates::Int64,
    ncontrols::Int64,
    init_val::Float64,
    distribution::Bool,
    transform_elements,
    comp_ids,
    n_par,
    m_par,
)

    # Initialize matrices for states and controls
    S_t = zeros(nstates, T)
    C_t = zeros(ncontrols, T)

    # Initial conditions: states by assumption, controls as implied by gx and initial state
    S_t[exovar, 1] = init_val
    C_t[:, 1] = gx * S_t[:, 1]

    # Simulation: iterate forward
    for t = 2:T
        S_t[:, t] = hx * S_t[:, t - 1]
        C_t[:, t] = gx * S_t[:, t]
    end

    # Recompute levels for the original IRFs, as defined in macro @generate_equations
    original = [S_t; C_t]
    level = fill(NaN64, size(original))

    # Start with the aggregate variables
    idx = [getfield(ids, Symbol(j)) for j in aggr_names]
    idxSS = [getfield(ids, Symbol(j, "SS")) for j in aggr_names]
    level[idx, :] = exp.(XSS[idxSS] .+ original[idx, :])

    if distribution
        dist_results = compute_irfs_inner_distribution(
            original,
            XSS,
            ids,
            comp_ids,
            transform_elements,
            n_par,
            m_par,
        )
        return original, level, dist_results
    else
        return original, level
    end
end

"""
    compute_irfs_inner_distribution(X, XSS, ids, compressionIndexes, transform_elements, n_par, m_par)

Reconstructs the full, uncompressed impulse responses for the marginal value functions (Wb,
Wk) and the joint probability distribution (PDF) from their compressed representations.

This function takes the 1D compressed impulse responses for all model variables and unpacks
them into their full-dimensional form for each period of the IRF. It follows the precise
logic of the `Fsys` function to combine steady-state values, transformation matrices (DCT
and Shuffle), and the IRF deviations to compute the dynamics of the entire state space.

# Arguments

  - `X::Array{Float64,2}`: A matrix `(n_vars x T)` containing the compressed IRFs for
    all variables.
  - `XSS::Vector{Float64}`: The steady-state vector of the model.
  - `ids`: An object mapping variable names to their indices in `X` and `XSS`.
  - `compressionIndexes`: An object containing the indices for compressed variables.
  - `transform_elements`: An object containing the transformation matrices (DCT, IDCT,
    Γ) needed for unpacking.
  - `n_par`: A struct with numerical parameters (e.g., grid sizes).
  - `m_par`: A struct with model parameters.

# Returns

  - `PDF::Array{Float64,4}`: The uncompressed IRF for the full joint probability
    distribution.
  - `Wb::Array{Float64,4}`, `Wk::Array{Float64,4}`: The uncompressed IRF for the marginal
    utility with respect to liquid/illiquid assets.
  - `PDF_b::Array{Float64,2}`, `PDF_k::Array{Float64,2}`, `PDF_h::Array{Float64,2}`: The
    uncompressed IRFs for the marginal PDFs of liquid assets, illiquid assets, and human
    capital, respectively.
  - `PDF_bk::Array{Float64,3}`, `PDF_bh::Array{Float64,3}`, `PDF_kh::Array{Float64,3}`: The
    uncompressed IRFs for the joint PDFs of liquid and illiquid assets, liquid assets and
    human capital, and illiquid assets and human capital, respectively.
  - `Wb_b::Array{Float64,2}`, `Wb_k::Array{Float64,2}`, `Wb_h::Array{Float64,2}`,
    `Wk_b::Array{Float64,2}`, `Wk_k::Array{Float64,2}`, `Wk_h::Array{Float64,2}`: The
    uncompressed IRFs for the marginal value functions with respect to liquid/illiquid
    assets, aggregated over the other two dimensions.
  - `Wb_bk::Array{Float64,3}`, `Wb_bh::Array{Float64,3}`, `Wb_kh::Array{Float64,3}`,
    `Wk_bk::Array{Float64,3}`, `Wk_bh::Array{Float64,3}`, `Wk_kh::Array{Float64,3}`: The
    uncompressed IRFs for the marginal value functions with respect to liquid/illiquid
    assets, aggregated over one dimension.
"""
function compute_irfs_inner_distribution(
    X::AbstractMatrix,
    XSS::AbstractVector,
    ids,
    compressionIndexes::Vector,
    transform_elements::TransformationElements,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    ## 1. Preamble & Setup ---------------------------------------------------------------
    nb, nk, nh = n_par.nb, n_par.nk, n_par.nh
    T = size(X, 2)

    ## 2. Getting IDs and unpacking Steady State Objects ---------------------------------
    vf_indexes_ss = ids.valueFunctionSS
    distr_indexes_ss = ids.distrSS

    vf_indexes = ids.valueFunction
    distr_indexes = ids.distr

    # Steady-state value functions (in log-inverse-marginal-utility space)
    vfSS = unpack_ss_valuefunctions(XSS, vf_indexes_ss, m_par, n_par)

    # Steady-state joint distribution object
    distrSS = unpack_ss_distributions(XSS, distr_indexes_ss, n_par)

    ## 3. Initialize Output Containers ---------------------------------------------------
    Wb = zeros(nb, nk, nh, T)   # marginal value of b (or m.u. of c wrt b)
    Wk = zeros(nb, nk, nh, T)   # marginal value of k
    PDF = zeros(nb, nk, nh, T)   # joint PDF(b,k,h)

    ## 4. Main Loop: Reconstruct Full Dynamics for Each Period --------------------------
    for t = 1:T
        X_t = view(X, :, t)

        # 4a) Value functions -----------------------------------------------------------
        vf_t = unpack_perturbed_valuefunctions(
            X_t,
            vf_indexes,
            vfSS,
            transform_elements,
            compressionIndexes,
            n_par,
        )

        Wb[:, :, :, t] = vf_t.b
        if n_par.model isa TwoAsset
            Wk[:, :, :, t] = vf_t.k
        end

        # 4b) Joint distribution via copula --------------------------------------------
        distr_t, _ = unpack_perturbed_distributions(
            X_t,
            X_t,                  # we don't care about the "prime" distribution here
            distrSS,
            distr_indexes,
            compressionIndexes,
            transform_elements,
            n_par,
        )
        CDF_joint = distr_t.COP
        PDF[:, :, :, t] = cdf_to_pdf(CDF_joint)
    end

    ## 5. Marginals and Output Dict -----------------------------------------------------

    all_marginals = Dict{String,Any}()
    merge!(all_marginals, compute_all_marginals(PDF, "PDF"))
    merge!(all_marginals, compute_all_marginals(Wb, "Wb"))

    if n_par.model isa TwoAsset
        # Only meaningful in the two-asset case
        merge!(all_marginals, compute_all_marginals(Wk, "Wk"))
    end

    # Also store full 3D×time objects
    all_marginals["PDF"] = PDF
    all_marginals["Wb"] = Wb
    if n_par.model isa TwoAsset
        all_marginals["Wk"] = Wk
    end

    return all_marginals
end

"""
    compute_all_marginals(
        Base_Array::AbstractArray{Float64,4},
        Base_Name::String
    )

Computes all 1D and 2D marginal distributions of a 4D array by summing over the appropriate
dimensions.

# Arguments

  - `Base_Array::AbstractArray{Float64,4}`: A 4-dimensional array with dimensions `(nb, nk, nh, T)`.
  - `Base_Name::String`: A string prefix ("PDF" or "W") used to generate the keys in the
    output dictionary.

# Returns

  - `Dict{String, AbstractArray}`: A dictionary mapping descriptive names to the computed
    marginal arrays. For example, for `Base_Name = "PDF"`, keys include:

      + `"PDF_b"`: Marginal over k and h (size: `(nb, T)`)
      + `"PDF_k"`: Marginal over b and h (size: `(nk, T)`)
      + `"PDF_h"`: Marginal over b and k (size: `(nh, T)`)
      + `"PDF_bk"`: Joint over b and k (size: `(nb, nk, T)`)
      + `"PDF_bh"`: Joint over b and h (size: `(nb, nh, T)`)
      + `"PDF_kh"`: Joint over k and h (size: `(nk, nh, T)`)

# Notes

  - The function sums over the unspecified dimensions to create the marginals (e.g., for
    `_b`, it sums over dimensions 2 and 3).
"""
function compute_all_marginals(Base_Array::AbstractArray{Float64,4}, Base_Name::String)
    # Base_Array is a 4D array: (nb, nk, nh, T)

    # Store results in a NamedTuple or Dict to be merged later
    marginals = Dict{String,AbstractArray}()

    # Mapping from dimension set to the dimensions to sum over (The dims in 'sum' are the
    # dimensions to get rid of)
    dim_map = (
        b = (2, 3), # sum over k and h
        k = (1, 3), # sum over b and h
        h = (1, 2), # sum over b and k
        bk = (3,),  # sum over h
        bh = (2,),  # sum over k
        kh = (1,),  # sum over b
    )

    for (dim_symbol, dims_to_sum) in pairs(dim_map)
        # 1. Compute the marginal/joint sum
        marginal_array = dropdims(sum(Base_Array; dims = dims_to_sum); dims = dims_to_sum)

        # 2. Store with the appropriate symbol (e.g., "PDF_b", "Wk_kh")
        field_name = string(Base_Name, "_", dim_symbol)
        marginals[field_name] = marginal_array
    end

    return marginals
end

"""
    compute_irf_confidence_intervals(
        draws_after_burnin::Matrix{Float64},
        exovar_idx::Int,
        init_val::Float64,
        n_vars::Int64,
        T::Int64,
        irf_interval_options
    )

Computes bootstrap-style confidence intervals for IRFs by repeatedly simulating them with
parameter vectors sampled from a posterior distribution.

# Arguments

  - `draws_after_burnin::Matrix{Float64}`: Matrix of posterior draws where each row is a
    parameter vector.

  - `exovar_idx::Int`: Positional index of the exogenous variable being shocked.
  - `init_val::Float64}`: Initial value (magnitude) of the shock.
  - `n_vars::Int64`: Number of endogenous variables in the system.
  - `T::Int64`: The time horizon for the IRFs.
  - `irf_interval_options`: A dictionary or other container holding options, including:

      + `n_replic::Int`: Number of bootstrap replications to run.
      + `percentile_bounds::Vector{Float64}`: A 2-element vector with the lower and upper
        percentile bounds (e.g., `[0.05, 0.95]`).

# Returns

  - `IRF_lower_bound::Matrix{Float64}`: Lower bound of the confidence interval for the IRF
    in deviations.
  - `IRF_upper_bound::Matrix{Float64}`: Upper bound of the confidence interval for the IRF
    in deviations.
  - `IRF_lvl_lower_bound::Matrix{Float64}`: Lower bound for the IRF in levels.
  - `IRF_lvl_upper_bound::Matrix{Float64}`: Upper bound for the IRF in levels.
"""
function compute_irf_confidence_intervals(
    draws_after_burnin::Matrix{Float64},
    exovar_idx::Int,
    init_val::Float64,
    n_vars::Int64,
    T::Int64,
    irf_interval_options,
)

    # Params
    n_replic = irf_interval_options["n_replic"]
    percentile_bounds = irf_interval_options["percentile_bounds"]

    # Draws
    rand_draws = rand(1:size(draws_after_burnin, 1), n_replic)

    # Pre-allocate
    IRF_draws_3d = Array{Float64,3}(undef, n_vars, T, n_replic)
    IRF_draws_lvl_3d = Array{Float64,3}(undef, n_vars, T, n_replic)

    # Loop through the rest of the sampled draws and compute IRFs
    for j = 1:n_replic
        idx = rand_draws[j]
        param_draw = draws_after_burnin[idx, :]

        # Compute the IRF for this specific parameter draw
        IRF_draws_3d[:, :, j], IRF_draws_lvl_3d[:, :, j] = compute_irf_given_param_draw(
            exovar_idx,
            init_val,
            param_draw,
            T,
            irf_interval_options,
        )
    end

    # Compute the quantiles
    IRF_lower_bound = compute_nanquantile_matrix(IRF_draws_3d, percentile_bounds[1])
    IRF_lvl_lower_bound = compute_nanquantile_matrix(IRF_draws_lvl_3d, percentile_bounds[1])
    IRF_upper_bound = compute_nanquantile_matrix(IRF_draws_3d, percentile_bounds[2])
    IRF_lvl_upper_bound = compute_nanquantile_matrix(IRF_draws_lvl_3d, percentile_bounds[2])
    # 6. Package and return the results
    return IRF_lower_bound, IRF_upper_bound, IRF_lvl_lower_bound, IRF_lvl_upper_bound
end

"""
    compute_nanquantile_matrix(b::AbstractArray{Float64, 3}, p::Real)

Computes the quantile for each vector slice of a 3D array along the third dimension,
robustly handling `NaN` values.

# Arguments

  - `b::AbstractArray{Float64, 3}`: A 3D array of data, typically with dimensions
    `(variables, time_periods, draws)`.
  - `p::Real`: The quantile to compute (e.g., 0.05 for the 5th percentile).

# Returns

  - `w::Matrix{Float64}`: A 2D matrix of shape `(variables, time_periods)` containing the
    computed quantiles for each slice.

# Notes

  - If all values in a vector slice `b[i, j, :]` are `NaN`, the resulting quantile for that
    element `w[i, j]` will also be `NaN`.
"""
function compute_nanquantile_matrix(b, p)
    w = Matrix{Float64}(undef, size(b, 1), size(b, 2))
    # 2. Loop through each column of the matrix
    for j in axes(b, 2)
        # 3. Loop through each row of the matrix
        for i in axes(b, 1)
            # For the current (i, j) position, grab the vector of all draws across the 3rd
            # dimension
            slice_of_draws = b[i, j, :]

            # Filter out any NaN values from that vector
            filtered_slice = filter(!isnan, slice_of_draws)

            # Check if the filtered vector is now empty. If it is, the result is NaN.
            # Otherwise, calculate the quantile.
            if isempty(filtered_slice)
                result = NaN
            else
                result = quantile(filtered_slice, p)
            end

            # Store the final result in our output matrix
            w[i, j] = result
        end
    end
    return w
end

"""
    compute_irf_given_param_draw(
        exovar,
        init_val,
        param_draw,
        T,
        irf_interval_options
    )

Computes the impulse response function (IRF) for a single draw of model parameters.

This is a helper function typically used within a bootstrap or MCMC loop for generating
confidence intervals.

# Arguments

  - `exovar::Int`: Index of the exogenous variable receiving the shock.

  - `init_val::Float64`: Initial value (magnitude) of the shock.
  - `param_draw::Vector{Float64}`: A vector containing a single draw of the model's free
    parameters.
  - `T::Int`: The time horizon for the IRF.
  - `irf_interval_options`: A dictionary or other container with required objects:

      + `"sr"`: The steady-state results object.
      + `"lr"`: The linear results object from the model solution.
      + `"m_par"`: A `Flatten` object for reconstructing parameter structs.
      + `"e_set"`: A settings object, especially for handling measurement error.

# Returns

  - A tuple `(IRFs, IRFs_lvl)`:

      + `IRFs::Matrix{Float64}`: The IRF in deviations from steady state.
      + `IRFs_lvl::Matrix{Float64}`: The IRF in levels.

# Notes

  - This function creates a local copy of the linear results (`lr`) before calling
    `update_model` to avoid mutating the original object.
  - Distributional IRFs are currently hardcoded to `false` within this function.
"""
function compute_irf_given_param_draw(exovar, init_val, param_draw, T, irf_interval_options)
    sr = irf_interval_options["sr"]
    lr = irf_interval_options["lr"]
    m_par = irf_interval_options["m_par"]
    e_set = irf_interval_options["e_set"]

    # Reconstruct model parameters
    if e_set.me_treatment != :fixed
        m_par_local = Flatten.reconstruct(
            m_par,
            param_draw[1:(length(param_draw) - length(e_set.meas_error_input))],
        )
    else
        m_par_local = Flatten.reconstruct(m_par, param_draw)
    end

    # Necessary since update_model modifies lr in-place
    A = lr.A
    B = lr.B
    State2Control = lr.State2Control
    LOMstate = lr.LOMstate
    SolutionError = lr.SolutionError
    nk = lr.nk

    lr_local = LinearResults(
        copy(State2Control),
        copy(LOMstate),
        copy(A),
        copy(B),
        copy(SolutionError),
        copy(nk),
    )

    lr_reduc = update_model(sr, lr_local, m_par_local)
    nstates = size(lr_reduc.LOMstate, 1)
    ncontrols = size(lr_reduc.State2Control, 1)

    # TODO: Hardcoded, assuming not matching distributional IRFs
    distribution = false
    comp_ids = nothing
    n_par = nothing
    m_par = nothing
    transform_elements = nothing

    IRFs, IRFs_lvl = compute_irfs_inner(
        exovar,
        lr_reduc.State2Control,
        lr_reduc.LOMstate,
        sr.XSS,
        sr.indexes_r,
        T,
        nstates,
        ncontrols,
        init_val,
        distribution,
        transform_elements,
        comp_ids,
        n_par,
        m_par,
    )
    return IRFs, IRFs_lvl
end
