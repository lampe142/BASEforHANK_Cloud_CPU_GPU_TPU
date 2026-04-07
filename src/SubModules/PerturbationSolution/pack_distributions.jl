"""
    unpack_ss_distributions(XSS, indexes_distrSS, n_par)

Unpack steady state distributions from the steady state vector `XSS` using the provided
`indexes`.

This function extracts the steady state distribution components (e.g., copula, marginals)
from the flat state vector `XSS` and organizes them into a distribution object (e.g.,
`CopulaTwoAssets`, `CopulaOneAsset`, `CDF`, `RepAgent`) based on the model type.

# Arguments

  - `XSS::Array{Float64,1}`: The steady state vector.
  - `indexes_distrSS`: An index structure (e.g., `CopulaTwoAssetsIndexes`) that maps
    distribution components to their locations in `XSS`.
  - `n_par::NumericalParameters`: Numerical parameters of the model.

# Returns

  - A distribution object (e.g., `CopulaTwoAssets`, `CopulaOneAsset`, `CDF`, `RepAgent`)
    containing the unpacked steady state distribution.
"""
function unpack_ss_distributions(
    XSS::Array{Float64,1},
    indexes_distrSS::CopulaTwoAssetsIndexes,
    n_par::NumericalParameters,
)
    return set_copula(
        reshape(XSS[indexes_distrSS.COP], (n_par.nb, n_par.nk, n_par.nh)),
        XSS[indexes_distrSS.b],
        XSS[indexes_distrSS.k],
        XSS[indexes_distrSS.h],
        n_par.transition_type,
    )
end

function unpack_ss_distributions(
    XSS::AbstractArray,
    indexes_distrSS::CopulaOneAssetIndexes,
    n_par::NumericalParameters,
)
    return set_copula(
        reshape(XSS[indexes_distrSS.COP], (n_par.nb, n_par.nh)),
        XSS[indexes_distrSS.b],
        XSS[indexes_distrSS.h],
        n_par.transition_type,
    )
end

function unpack_ss_distributions(
    XSS::AbstractArray,
    indexes_distrSS::CDFIndexes,
    n_par::NumericalParameters,
)
    return CDF(reshape(XSS[indexes_distrSS.CDF], (n_par.nb, n_par.nh)))
end

function unpack_ss_distributions(
    XSS::AbstractArray,
    indexes_distrSS::RepAgentIndexes,
    n_par::NumericalParameters,
)
    return RepAgent(reshape(XSS[indexes_distrSS.h], (n_par.nh)))
end

"""
    unpack_COP_dev(X, indexes_distr, compressionIndexes, transform_elements, n_par)

Unpack perturbed copula deviations from the state vector `X`.

This function extracts the compressed copula deviations (DCT coefficients) from the state
vector `X`, uncompresses them using the inverse DCT, and reshapes them into the full grid
dimensions. It returns the deviations in CDF space.

# Arguments

  - `X::AbstractArray`: The state vector containing perturbed variables.
  - `indexes_distr`: Index structure for the distribution variables.
  - `compressionIndexes`: Vector of indices for the retained DCT coefficients.
  - `transform_elements`: Transformation matrices struct (including `DCD`, `IDCD`).
  - `n_par::NumericalParameters`: Numerical parameters.

# Returns

  - `COP_Dev`: Array representing the perturbed copula deviations in CDF space.
"""
function unpack_COP_dev(
    X::AbstractArray,
    indexes_distr::Union{CopulaTwoAssetsIndexes,CopulaOneAssetIndexes},
    compressionIndexes::Vector,
    transform_elements::TransformationElements,
    n_par::NumericalParameters,
)
    n_dims = shape_COP_dev(n_par, n_par.model)
    θD = uncompress(
        compressionIndexes[2],
        X[indexes_distr.COP],
        transform_elements.DCD,
        transform_elements.IDCD,
        n_par.model,
    )
    COP_Dev = reshape(copy(θD[:]), n_dims)
    COP_Dev = pdf_to_cdf(COP_Dev)
    return COP_Dev
end

shape_COP_dev(n_par::NumericalParameters, model::OneAsset) =
    (n_par.nb_copula, n_par.nh_copula)
shape_COP_dev(n_par::NumericalParameters, model::TwoAsset) =
    (n_par.nb_copula, n_par.nk_copula, n_par.nh_copula)

"""
    unpack_perturbed_distributions(X, XPrime, distrSS, indexes_distr, compressionIndexes, transform_elements, n_par)

Unpack perturbed distributions from the state vectors `X` and `XPrime`.

This function reconstructs the full distribution objects for the current period (`X`) and
the next period (`XPrime`) by adding the unpacked deviations to the steady state
distribution `distrSS`. It handles both marginal distributions (using `Γ` matrices) and the
copula (using DCT uncompression).

# Arguments

  - `X`, `XPrime`: State vectors for current and next periods.
  - `distrSS`: Steady state distribution object.
  - `indexes_distr`: Index structure for distribution variables.
  - `compressionIndexes`: Indices for retained DCT coefficients.
  - `transform_elements`: Transformation matrices struct.
  - `n_par::NumericalParameters`: Numerical parameters.

# Returns

  - A tuple of two distribution objects: one for period `t` and one for period `t+1`.
"""
function unpack_perturbed_distributions(
    X::AbstractArray,
    XPrime::AbstractArray,
    distrSS::CopulaOneAsset,
    indexes_distr::CopulaOneAssetIndexes,
    compressionIndexes::Vector,
    transform_elements::TransformationElements,
    n_par::NumericalParameters,
)
    dt = typeof(distrSS) # distribution type
    tt = n_par.transition_type

    # Unpack perturbed copula deviations
    COP_Dev =
        unpack_COP_dev(X, indexes_distr, compressionIndexes, transform_elements, n_par)
    COP_DevPrime =
        unpack_COP_dev(XPrime, indexes_distr, compressionIndexes, transform_elements, n_par)

    # REVIEW - move outside function?
    get_perturbed_marginals(
        x,
        xPrime,
        dSS::CopulaPDFsOneAsset,
        i::Int64,
        a::Symbol,
        transftype::Union{LinearTransformation,ParetoTransformation},
    ) = getfield(dSS, a) .+ transform_elements.Γ[i] * x[getfield(indexes_distr, a)],
    getfield(dSS, a) .+ transform_elements.Γ[i] * xPrime[getfield(indexes_distr, a)]
    get_perturbed_marginals(
        x,
        xPrime,
        dSS::CopulaCDFsOneAsset,
        i::Int64,
        a::Symbol,
        transftype::LinearTransformation,
    ) = getfield(dSS, a) .+ [x[getfield(indexes_distr, a)]; 0],
    getfield(dSS, a) .+ [xPrime[getfield(indexes_distr, a)]; 0]
    function get_perturbed_marginals(
        x,
        xPrime,
        dSS::CopulaCDFsOneAsset,
        i::Int64,
        a::Symbol,
        transftype::ParetoTransformation,
    )
        CDF_mSS = getfield(dSS, a)
        X_CDF_m = x[getfield(indexes_distr, a)]
        CDF_m = zeros(eltype(x), n_par.nb)
        CDF_m[1] = CDF_mSS[1] * exp.(X_CDF_m[1])
        alpSS_2 = -log.((1.0 .- CDF_mSS[2]) / (1.0 .- CDF_mSS[1]))
        CDF_m[2] = 1.0 - (1.0 .- CDF_m[1]) .* exp.(-alpSS_2 * exp.(X_CDF_m[2]))
        for i = 3:(n_par.nb - 1)
            alpSS =
                -log.((1.0 .- CDF_mSS[i]) / (1.0 .- CDF_mSS[i - 1])) /
                log.(n_par.grid_b[i] / n_par.grid_b[i - 1])
            CDF_m[i] =
                1.0 -
                (1.0 .- CDF_m[i - 1]) .*
                exp.(-alpSS * exp.(X_CDF_m[i]) * log(n_par.grid_b[i] / n_par.grid_b[i - 1]))
        end
        CDF_m[end] = CDF_mSS[end]

        XPrime_CDF_m = xPrime[getfield(indexes_distr, a)]
        CDF_m_Prime_dev = zeros(eltype(xPrime), n_par.nb - 1)
        CDF_m_Prime_dev[1] = XPrime_CDF_m[1] .+ log.(CDF_mSS[1])
        CDF_m_Prime_dev[2] = XPrime_CDF_m[2] .+ log(alpSS_2)
        for i = 3:(n_par.nb - 1)
            alpSS =
                -log.((1 - CDF_mSS[i]) / (1 - CDF_mSS[i - 1])) /
                log.(n_par.grid_b[i] / n_par.grid_b[i - 1])
            CDF_m_Prime_dev[i] = XPrime_CDF_m[i] .+ log(alpSS)
        end

        return CDF_m, CDF_m_Prime_dev
    end

    # Perturbed distributions based on marginal CDFs as states
    distr_b, distr_b_Prime =
        get_perturbed_marginals(X, XPrime, distrSS, 1, :b, n_par.transf_CDF)
    # Treat income distribution as discrete
    distr_h = get_PDF(distrSS.h, dt) .+ transform_elements.Γ[2] * X[indexes_distr.h]
    distr_h_Prime =
        get_PDF(distrSS.h, dt) .+ transform_elements.Γ[2] * XPrime[indexes_distr.h]
    # Non-linear transition: store CDFs
    if isa(tt, NonLinearTransition)
        distr_h = pdf_to_cdf(distr_h)
        distr_h_Prime = pdf_to_cdf(distr_h_Prime)
    end

    # Steady state copula marginals (cdfs)
    s_m_b = n_par.copula_marginal_b .+ zeros(eltype(X), 1)
    s_m_h = n_par.copula_marginal_h .+ zeros(eltype(X), 1)

    ## Joint distribution -----------------------------------------------------------------
    CDF_joint = Copula(
        get_CDF(distr_b[:], dt),
        get_CDF(distr_h[:], dt),
        distrSS,
        COP_Dev,
        s_m_b,
        s_m_h,
        tt,
    )

    return set_copula(CDF_joint, distr_b, distr_h, tt),
    set_copula(COP_DevPrime, distr_b_Prime, distr_h_Prime, tt)
end

function Copula(
    x::Vector,
    z::Vector,
    distrSS::CopulaPDFsOneAsset,
    COP_Dev,
    s_m_b,
    s_m_h,
    ::LinearTransition,
)
    return mylinearinterpolate2(
        get_CDF(distrSS.b, typeof(distrSS)),
        get_CDF(distrSS.h, typeof(distrSS)),
        distrSS.COP,
        x,
        z,
    ) .+ mylinearinterpolate2(s_m_b, s_m_h, COP_Dev, x, z)
end

function Copula(
    x::Vector,
    z::Vector,
    distrSS::CopulaCDFsOneAsset,
    COP_Dev,
    s_m_b,
    s_m_h,
    ::NonLinearTransition,
)
    CDF = zeros(eltype(s_m_b), size(distrSS.COP))
    COP =
        distrSS.COP .+
        marginal_to_joint(s_m_b, COP_Dev, distrSS.b; monotonic_spline = false)
    CDF .= marginal_to_joint(distrSS.b, COP, x)
    return CDF
end

function unpack_perturbed_distributions(
    X::AbstractArray,
    XPrime::AbstractArray,
    distrSS::CopulaTwoAssets,
    indexes_distr::CopulaTwoAssetsIndexes,
    compressionIndexes::Vector,
    transform_elements::TransformationElements,
    n_par::NumericalParameters,
)
    dt = typeof(distrSS)
    tt = n_par.transition_type

    COP_Dev =
        unpack_COP_dev(X, indexes_distr, compressionIndexes, transform_elements, n_par)
    COP_DevPrime =
        unpack_COP_dev(XPrime, indexes_distr, compressionIndexes, transform_elements, n_par)

    get_perturbed_marginal(x, dSS::CopulaPDFsTwoAssets, i::Int64, a::Symbol) =
        getfield(dSS, a) .+ transform_elements.Γ[i] * x[getfield(indexes_distr, a)]
    get_perturbed_marginal(x, dSS::CopulaCDFsTwoAssets, i::Int64, a::Symbol) =
        getfield(dSS, a) .+ [x[getfield(indexes_distr, a)]; 0]

    distr_b = get_perturbed_marginal(X, distrSS, 1, :b)
    distr_k = get_perturbed_marginal(X, distrSS, 2, :k)
    # Treat income distribution always as discrete
    distr_h = distrSS.h .+ transform_elements.Γ[3] * X[indexes_distr.h]

    distr_b_Prime = get_perturbed_marginal(XPrime, distrSS, 1, :b)
    distr_k_Prime = get_perturbed_marginal(XPrime, distrSS, 2, :k)
    distr_h_Prime = distrSS.h .+ transform_elements.Γ[3] * XPrime[indexes_distr.h]

    # steady state copula marginals (cdfs)
    s_m_b = n_par.copula_marginal_b .+ zeros(eltype(X), 1)
    s_m_k = n_par.copula_marginal_k .+ zeros(eltype(X), 1)
    s_m_h = n_par.copula_marginal_h .+ zeros(eltype(X), 1)

    ## Joint distribution -----------------------------------------------------------------
    Copula(x::Vector, y::Vector, z::Vector) =
        myinterpolate3(
            get_CDF(distrSS.b, dt),
            get_CDF(distrSS.k, dt),
            get_CDF(distrSS.h, dt),
            distrSS.COP,
            n_par.model,
            x,
            y,
            z,
        ) .+ myinterpolate3(s_m_b, s_m_k, s_m_h, COP_Dev, n_par.model, x, y, z)

    CDF_joint =
        Copula(get_CDF(distr_b[:], dt), get_CDF(distr_k[:], dt), get_CDF(distr_h[:], dt))

    return set_copula(CDF_joint, distr_b, distr_k, distr_h, tt),
    set_copula(COP_DevPrime, distr_b_Prime, distr_k_Prime, distr_h_Prime, tt)
end

function unpack_perturbed_distributions(
    X::AbstractArray,
    XPrime::AbstractArray,
    CDFsSS::CDF,
    indexes_distr::CDFIndexes,
    compressionIndexes::Vector,
    transform_elements::TransformationElements,
    n_par::NumericalParameters,
)
    # Back out perturbed CDF from deviations
    set_CDFs(x, xPrime, transftype::LinearTransformation) =
        CDF(CDF_joint_from_X(x, n_par, indexes_distr, CDFsSS.CDF, transform_elements.Γ[1])),
        CDF(
            CDF_joint_from_X(
                xPrime,
                n_par,
                indexes_distr,
                CDFsSS.CDF,
                transform_elements.Γ[1],
            ),
        )

    function set_CDFs(x, xPrime, transftype::ParetoTransformation)
        # Convert to conditional CDF
        CDF_jointSS_cond = copy(CDFsSS.CDF)
        CDF_jointSS_cond[:, 2:end] .= diff(CDF_jointSS_cond; dims = 2)
        PDF_ySS = CDF_jointSS_cond[end, :][:]

        X_CDF = reshape([x[indexes_distr.CDF]; 0.0], (n_par.nb, n_par.nh))    # the added 0.0 is replaced later
        CDF_joint = zeros(eltype(x), n_par.nb, n_par.nh)

        XPrime_CDF = reshape([xPrime[indexes_distr.CDF]; 0.0], (n_par.nb, n_par.nh))
        CDF_Prime_dev = zeros(eltype(x), n_par.nb, n_par.nh)

        # TODO: restructure this such that time-varying PDF_y is possible, would need separate saving of PDF_y?
        for i_h = 1:(n_par.nh)
            idx_start_pareto = transform_elements.pareto_indices[1][i_h] + 1
            CDF_joint[1:idx_start_pareto, i_h] =
                CDF_jointSS_cond[1:idx_start_pareto, i_h] .+ X_CDF[1:idx_start_pareto, i_h]
            CDF_Prime_dev[1:idx_start_pareto, i_h] =
                CDF_jointSS_cond[1:idx_start_pareto, i_h] .+
                XPrime_CDF[1:idx_start_pareto, i_h]
            idx_end_pareto = transform_elements.pareto_indices[2][i_h]
            for i = (idx_start_pareto + 1):(idx_end_pareto - 1)
                alpSS =
                    -log.(
                        (PDF_ySS[i_h] .- CDF_jointSS_cond[i, i_h]) ./
                        (PDF_ySS[i_h] .- CDF_jointSS_cond[i - 1, i_h])
                    ) ./ log.(n_par.grid_b[i] / n_par.grid_b[i - 1])
                CDF_joint[i, i_h] =
                    PDF_ySS[i_h] -
                    (PDF_ySS[i_h] .- CDF_joint[i - 1, i_h]) .*
                    exp.(
                        -alpSS .* exp.(X_CDF[i, i_h]) .*
                        log(n_par.grid_b[i] / n_par.grid_b[i - 1])
                    )
                (alpSS == 0.0) &&
                    (@warn "alpSS = $alpSS, so log(alpSS) is -Inf for i_b = $i and i_h = $i_h")
                CDF_Prime_dev[i, i_h] = XPrime_CDF[i, i_h] .+ log.(alpSS)
            end
            CDF_joint[idx_end_pareto:(end - 1), i_h] =
                CDF_jointSS_cond[idx_end_pareto:(end - 1), i_h] .+
                X_CDF[idx_end_pareto:(end - 1), i_h]
            CDF_joint[end, i_h] = PDF_ySS[i_h] .+ X_CDF[end, i_h]
            CDF_Prime_dev[idx_end_pareto:(end - 1), i_h] =
                CDF_jointSS_cond[idx_end_pareto:(end - 1), i_h] .+
                XPrime_CDF[idx_end_pareto:(end - 1), i_h]
            CDF_Prime_dev[end, i_h] = PDF_ySS[i_h] .+ XPrime_CDF[end, i_h]
        end
        return CDF(cumsum(CDF_joint; dims = 2)), CDF(CDF_Prime_dev)
    end
    return set_CDFs(X, XPrime, n_par.transf_CDF)
end

function unpack_perturbed_distributions(
    X::AbstractArray,
    XPrime::AbstractArray,
    CDFsSS::RepAgent,
    indexes_distr::RepAgentIndexes,
    compressionIndexes::Vector,
    transform_elements::TransformationElements,
    n_par::NumericalParameters,
)
    # Back out perturbed CDF from deviations
    distr_h = CDFsSS.h .+ cumsum(transform_elements.Γ[1] * X[indexes_distr.h])
    distr_hPrime = CDFsSS.h .+ cumsum(transform_elements.Γ[1] * XPrime[indexes_distr.h])
    return RepAgent(distr_h), RepAgent(distr_hPrime)
end

"""
    marginal_to_joint(x_grid, y_grid, x; monotonic_spline=true)

Compute a joint cumulative distribution function (CDF) from marginal distributions using
spline interpolation.

This function interpolates the marginal distributions (`y_grid` on `x_grid`) to the new
evaluation points `x`.

# Arguments

  - `x_grid::AbstractArray`: The grid points for the marginal distributions.
  - `y_grid::AbstractMatrix`: The marginal distributions corresponding to `x_grid`.
  - `x::AbstractArray`: The evaluation points for the joint CDF.
  - `monotonic_spline::Bool` (default `true`): If `true`, use monotonic spline
    interpolation; otherwise, use cubic Hermite splines.

# Returns

  - `CDF_j::AbstractMatrix`: The joint CDF evaluated at the points in `x`.
"""
function marginal_to_joint(
    x_grid::AbstractArray,
    y_grid::AbstractMatrix,
    x::AbstractArray;
    monotonic_spline = true,
)
    T = promote_type(eltype(x), eltype(x_grid), eltype(y_grid))
    CDF_j = zeros(T, size(x, 1), size(y_grid, 2))
    epsilon = one(T) * 1e-14
    for i_h in axes(y_grid, 2)
        if monotonic_spline
            marginal_to_joint_splines = Interpolator(x_grid, y_grid[:, i_h])
        else
            dydx_left = zeros(T, size(x_grid))
            dydx_right = zeros(T, size(x_grid))
            dydx_av = zeros(T, size(x_grid))
            for i_x = 2:(length(x_grid) - 1)
                dydx_left[i_x] =
                    (y_grid[i_x, i_h] - y_grid[i_x - 1, i_h]) /
                    (x_grid[i_x] - x_grid[i_x - 1])
                dydx_right[i_x] =
                    (y_grid[i_x + 1, i_h] - y_grid[i_x, i_h]) /
                    (x_grid[i_x + 1] - x_grid[i_x])
            end
            dydx_av[1] = dydx_right[2]
            dydx_av[end] = dydx_left[end - 1]
            dydx_av[2:(end - 1)] = 0.5 * (dydx_left[2:(end - 1)] + dydx_right[2:(end - 1)])
            marginal_to_joint_splines = Interpolator(x_grid, y_grid[:, i_h], dydx_av)
        end
        for i_x in eachindex(x)
            if x[i_x] < x_grid[1]
                dCDF_dx0 =
                    (
                        marginal_to_joint_splines(x_grid[1] + epsilon) -
                        marginal_to_joint_splines(x_grid[1] + epsilon * 0.0)
                    ) / epsilon
                CDF_j[i_x, i_h] = y_grid[1, i_h] + dCDF_dx0 * (x[1] - x_grid[1])
            elseif x[i_x] > x_grid[end]
                dCDF_dxN =
                    (
                        marginal_to_joint_splines(x_grid[end] + epsilon * 0.0) -
                        marginal_to_joint_splines(x_grid[end] - epsilon)
                    ) / epsilon
                CDF_j[i_x, i_h] = y_grid[end, i_h] + dCDF_dxN * (x[end] - x_grid[end])
            else
                CDF_j[i_x, i_h] = marginal_to_joint_splines(x[i_x])
            end
        end
    end
    return CDF_j
end

"""
    CDF_joint_from_X(X, n_par, indexes, CDF_jointSS, Γ)

Back out the joint CDF from the steady-state distribution and deviations `X`.

# Arguments

  - `X::AbstractArray`: Deviations from the steady-state distribution.
  - `n_par::NumericalParameters`: Numerical parameters of the model.
  - `indexes::IndexStruct`: Index structure for accessing elements in `X`.
  - `CDF_jointSS::AbstractArray`: Steady-state joint CDF.
  - `Γ::AbstractMatrix`: Transformation matrix for adjusting marginal distributions.

# Returns

  - `CDF_joint::AbstractArray`: The updated joint CDF after applying deviations.
"""
function CDF_joint_from_X(
    X::AbstractArray,
    n_par::NumericalParameters,
    indexes::CDFIndexes,
    CDF_jointSS::AbstractArray,
    Γ::AbstractMatrix,
)
    # Convert to conditional CDF
    CDF_jointSS_cond = copy(CDF_jointSS)
    CDF_jointSS_cond[:, 2:end] .= diff(CDF_jointSS_cond; dims = 2)

    X_CDF = reshape([X[indexes.CDF]; 0.0], (n_par.nb, n_par.nh))    # the added 0.0 is replaced later
    X_PDF_y = Γ * X_CDF[end, 1:(end - 1)] # take all but last element of PDF_y and apply shuffle matrix
    X_CDF[end, :] = X_PDF_y
    return cumsum(reshape(CDF_jointSS_cond, (n_par.nb, n_par.nh)) .+ X_CDF; dims = 2) # return CDF in both dimensions
end

"""
    unpack_ss_valuefunctions(XSS, indexes_vf, m_par, n_par)

Unpack steady state value functions from the state vector `XSS` using the provided
`indexes`.

This function reconstructs the value function object (e.g., `ValueFunctionsOneAsset`,
`ValueFunctionsTwoAssets`) from the flattened state vector `XSS`.

# Arguments

  - `XSS::Array{Float64,1}`: The steady state vector.
  - `indexes_vf`: Index structure for value function variables.
  - `m_par::ModelParameters`: Model parameters.
  - `n_par::NumericalParameters`: Numerical parameters.

# Returns

  - Value function object containing the steady state value functions.
"""
function unpack_ss_valuefunctions(
    XSS::Array{Float64,1},
    indexes_vf::ValueFunctionsOneAssetIndexes,
    m_par::ModelParameters,
    n_par::NumericalParameters,
)
    shape = (n_par.nb, n_par.nh)
    return ValueFunctionsOneAsset(reshape(XSS[indexes_vf.b], shape)) # return in log-inverse-marginal-utility space
end

function unpack_ss_valuefunctions(
    XSS::Array{Float64,1},
    indexes_vf::ValueFunctionsTwoAssetsIndexes,
    m_par::ModelParameters,
    n_par::NumericalParameters,
)
    shape = (n_par.nb, n_par.nk, n_par.nh)
    return ValueFunctionsTwoAssets(
        reshape(XSS[indexes_vf.b], shape),
        reshape(XSS[indexes_vf.k], shape),
    ) # return in log-inverse-marginal-utility space
end

function unpack_ss_valuefunctions(
    XSS::Array{Float64,1},
    indexes_vf::ValueFunctionsCompleteMarketsIndexes,
    m_par::ModelParameters,
    n_par::NumericalParameters,
)
    shape = (n_par.nb, n_par.nh)
    return ValueFunctionsCompleteMarkets(reshape(XSS[indexes_vf.b], shape)) # return in log-inverse-marginal-utility space
end

"""
    unpack_perturbed_valuefunctions(XPrime, indexes_vf, vfSS, compres, compressionIndexes, n_par)

Unpack perturbed value functions from the state vector `XPrime`.

This function reconstructs the full value functions for the next period by adding the
uncompressed deviations (from DCT coefficients in `XPrime`) to the steady state value
functions `vfSS`. The result is transformed back from log-inverse-marginal-utility space to
marginal utility.

# Arguments

  - `XPrime::AbstractArray`: The state vector for the next period.
  - `indexes_vf`: Index structure for value function variables.
  - `vfSS`: Steady state value functions.
  - `compres`: Transformation matrices struct.
  - `compressionIndexes`: Indices for retained DCT coefficients.
  - `n_par::NumericalParameters`: Numerical parameters.

# Returns

  - Value function object containing the perturbed value functions for the next period.
"""
function unpack_perturbed_valuefunctions(
    XPrime::AbstractArray,
    indexes_vf::ValueFunctionsOneAssetIndexes,
    vfSS::ValueFunctionsOneAsset,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
)
    vfPrime_b = mutil(
        exp.(
            vfSS.b[:] .+ uncompress(
                compressionIndexes[1][1],
                XPrime[indexes_vf.b],
                compres.DC,
                compres.IDC,
                n_par.model,
            )
        ),
        n_par.m_par,
    )
    return ValueFunctionsOneAsset(reshape(vfPrime_b, (n_par.nb, n_par.nh)))
end

function unpack_perturbed_valuefunctions(
    XPrime::AbstractArray,
    indexes_vf::ValueFunctionsTwoAssetsIndexes,
    vfSS::ValueFunctionsTwoAssets,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
)
    vfPrime_b = mutil(
        exp.(
            vfSS.b[:] .+ uncompress(
                compressionIndexes[1][1],
                XPrime[indexes_vf.b],
                compres.DC,
                compres.IDC,
                n_par.model,
            )
        ),
        n_par.m_par,
    )
    vfPrime_k = mutil(
        exp.(
            vfSS.k[:] .+ uncompress(
                compressionIndexes[1][2],
                XPrime[indexes_vf.k],
                compres.DC,
                compres.IDC,
                n_par.model,
            )
        ),
        n_par.m_par,
    )
    return ValueFunctionsTwoAssets(
        reshape(vfPrime_b, (n_par.nb, n_par.nk, n_par.nh)),
        reshape(vfPrime_k, (n_par.nb, n_par.nk, n_par.nh)),
    )
end

function unpack_perturbed_valuefunctions(
    XPrime::AbstractArray,
    indexes_vf::ValueFunctionsCompleteMarketsIndexes,
    vfSS::ValueFunctionsCompleteMarkets,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
)
    return ValueFunctionsCompleteMarkets(ones(eltype(XPrime), n_par.nb, n_par.nh))
end

"""
    unpack_transition_matrix(X, σ, n_par, m_par, model)

Construct the transition matrix `Π` for the income process, potentially including
perturbations.

# Arguments

  - `X`: State vector (used for type inference and potential state dependence).
  - `σ`: Volatility parameter (used if transition depends on volatility).
  - `n_par`: Numerical parameters.
  - `m_par`: Model parameters.
  - `model`: Model type (`OneAsset`, `TwoAsset`, `CompleteMarkets`).

# Returns

  - `Π`: The transition matrix.
"""
function unpack_transition_matrix(
    X::AbstractArray,
    σ,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    model::Union{OneAsset,TwoAsset},
)
    Π = n_par.Π .+ zeros(eltype(X), 1)[1]
    PP = ExTransition(m_par.ρ_h, n_par.bounds_h, sqrt(σ))
    if n_par.entrepreneur
        Π[1:(end - 1), 1:(end - 1)] = PP .* (1.0 - m_par.ζ)
    end
    return Π
end

function unpack_transition_matrix(
    X::AbstractArray,
    σ,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    model::CompleteMarkets,
)
    Π = n_par.Π .+ zeros(eltype(X), 1)[1]
    return Π
end
