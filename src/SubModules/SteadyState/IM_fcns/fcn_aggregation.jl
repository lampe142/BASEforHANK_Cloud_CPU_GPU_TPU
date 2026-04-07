## ------------------------------------------------------------------------------------
## Low-level integration and aggregation functions
## ------------------------------------------------------------------------------------

"""
    integrate_asset(cdf::AbstractVector, grid::AbstractVector, bound)

Compute aggregate asset holdings E[K] = ∫ k f(k) dk from a savings CDF using integration by
parts.

Arguments:

  - `cdf::AbstractVector`: cumulative distribution F(k) defined on `grid`
  - `grid::AbstractVector`: increasing asset grid
  - `bound`: optional upper integration bound; if `nothing` integrates over the full grid.
    Methods exist for `bound::Nothing` and `bound::Real` (or `Float64`).

Details:

  - Uses an interpolator for the CDF and computes E[K] = k*F(k) |_{k_min}^{bound} -
    ∫_{k_min}^{bound} F(k) dk
"""
function integrate_asset(cdf::AbstractVector, grid::AbstractVector, bound::Nothing)
    m_to_cdf_splines = Interpolator(grid, cdf)

    right_part = integrate(m_to_cdf_splines, grid[1], grid[end])
    left_part =
        grid[end] * m_to_cdf_splines(grid[end]) - grid[1] * m_to_cdf_splines(grid[1])
    E_K = left_part - right_part
    return E_K
end

function integrate_asset(cdf::AbstractVector, grid::AbstractVector, bound::Real)
    m_to_cdf_splines = Interpolator(grid, cdf)

    right_part = integrate(m_to_cdf_splines, grid[1], bound)
    left_part = bound * m_to_cdf_splines(bound) - grid[1] * m_to_cdf_splines(grid[1])
    E_K = left_part - right_part
    return E_K
end

integrate_asset(cdf::AbstractVector, grid::AbstractVector) =
    integrate_asset(cdf, grid, nothing)

"""
    aggregate(pdf::AbstractVector, grid::AbstractVector, bound)

Compute aggregate asset holdings by integrating over the PDF.

Arguments:

  - `pdf::AbstractVector`: Probability density function over asset grid
  - `grid::AbstractVector`: Asset grid
  - `bound`: Optional upper bound for integration; if `nothing` integrates over full grid.

Returns:

  - Aggregate asset holdings (dot product over grid or truncated grid)
"""
aggregate(pdf::AbstractVector, grid::AbstractVector, bound::Nothing) = dot(pdf, grid)

aggregate(pdf::AbstractVector, grid::AbstractVector, bound::Float64) =
    dot(pdf, (grid .< bound) .* grid)

## ------------------------------------------------------------------------------------
## Aggregation functions: wrappers
## ------------------------------------------------------------------------------------

"""
    aggregate_asset_helper(distr, grid, transition_type; bound=nothing, pdf_input=true)

Compute aggregate asset holdings from a marginal distribution for either linear or
nonlinear transition schemes.

Arguments:

  - `distr::AbstractVector`: marginal distribution values (PDF if `pdf_input=true`, CDF if `pdf_input=false`)

  - `grid::AbstractVector`: asset grid
  - `transition_type::LinearTransition | NonLinearTransition`: determines aggregation method

      + `LinearTransition`: uses pointwise aggregation over the grid.

          * If `distr` is a PDF, calls `aggregate(distr, grid, bound)`.
          * If `distr` is a CDF, converts to a PDF using `cdf_to_pdf(distr)` and calls `aggregate`.

      + `NonLinearTransition`: uses integration-by-parts on the CDF.

          * If `distr` is a PDF, first converts to a CDF via `cumsum(distr)` and calls `integrate_asset`.
          * If `distr` is a CDF, calls `integrate_asset` directly.
  - `bound`: optional upper integration bound (default `nothing`)
  - `pdf_input::Bool`: true if `distr` is a PDF, false if `distr` is a CDF

Returns:

  - Aggregate asset holdings (scalar)
"""
function aggregate_asset_helper(
    distr::AbstractVector,
    grid::AbstractVector,
    transition_type::LinearTransition,
    bound = nothing,
    pdf_input::Bool = true,
)
    return pdf_input ? aggregate(distr, grid, bound) :
           aggregate(cdf_to_pdf(distr), grid, bound)
end

function aggregate_asset_helper(
    distr::AbstractVector,
    grid::AbstractVector,
    transition_type::NonLinearTransition,
    bound = nothing,
    pdf_input::Bool = true,
)
    return pdf_input ? integrate_asset(cumsum(distr), grid, bound) :
           integrate_asset(distr, grid, bound)
end

## ------------------------------------------------------------------------------------
## Aggregation functions: interface
## ------------------------------------------------------------------------------------

"""
    aggregate_asset(distr, asset, n_par, bound=nothing)

Aggregate asset holdings across the distribution of agents.

Arguments

  - `distr::Union{CDF, CopulaOneAsset, CopulaTwoAssets, RepAgent}`: The distribution of
    agents
  - `asset::Symbol`: The asset to aggregate (`:b` for bonds, `:k` for capital, `:h` for
    productivity)
  - `n_par::NumericalParameters`: Numerical parameters containing asset grids (uses
    `grid_<asset>` and `transition_type`)
  - `bound`: optional upper integration bound (default `nothing` = full grid)

Returns

  - The aggregate level of the specified asset across all agents.
"""
function aggregate_asset(
    distr::CDF,
    asset::Symbol,
    n_par::NumericalParameters,
    bound = nothing,
)
    ndims_CDF = ndims(distr.CDF)
    cdf_asset = get_asset_distr(distr, asset)
    return aggregate_asset_helper(
        cdf_asset,
        getfield(n_par, Symbol("grid_", asset)),
        n_par.transition_type,
        bound,
        false,
    )
end

function aggregate_asset(
    distr::Union{CopulaOneAsset,CopulaTwoAssets},
    asset::Symbol,
    n_par::NumericalParameters,
    bound = nothing,
)
    is_pdf = isa(distr, CopulaPDFsOneAsset) || isa(distr, CopulaPDFsTwoAssets)
    return aggregate_asset_helper(
        getfield(distr, asset),
        getfield(n_par, Symbol("grid_", asset)),
        n_par.transition_type,
        bound,
        is_pdf,
    )
end

function aggregate_asset(
    distr::RepAgent,
    asset::Symbol,
    n_par::NumericalParameters,
    bound = nothing,
)
    return 1.0
end

## ------------------------------------------------------------------------------------
## Aggregation functions: B and K
## ------------------------------------------------------------------------------------

"""
    aggregate_B_K(distr, m_par, n_par, model)

Calculate aggregate asset holdings depending on the model type.

Arguments

  - `distr`: Distribution (unused for complete markets)
  - `m_par::ModelParameters`: Model parameters (contains β for interest rate calculation, ψ
    for portfolio share)
  - `n_par::NumericalParameters`: Numerical parameters (grids etc.)
  - `model::AbstractMacroModel`: Model type (CompleteMarkets, OneAsset, TwoAsset)

Returns

  - `(B, K)`: Aggregate bonds and aggregate capital (ordering as returned by each method)
"""
function aggregate_B_K(
    distr::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    model::CompleteMarkets,
)
    @assert @isdefined(CompMarketsCapital) "Complete Markets Model requires CompMarketsCapital function."
    rSS = (1.0 .- m_par.β) ./ m_par.β  # complete markets interest rate
    K = CompMarketsCapital(rSS, m_par)
    B = m_par.ψ * K
    return B, K
end

function aggregate_B_K(
    distr::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    model::OneAsset,
    pdf_input::Bool = true,
)
    supply_liquid = aggregate_asset_helper(
        sum(distr; dims = 2)[:],
        n_par.grid_b,
        n_par.transition_type,
        nothing,
        pdf_input,
    )
    supply_illiquid = (1.0 - m_par.ψ) * supply_liquid
    supply_liquid = m_par.ψ * supply_liquid
    return supply_liquid, supply_illiquid
end

function aggregate_B_K(
    distr::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    model::TwoAsset,
)
    supply_illiquid = aggregate_asset_helper(
        sum(distr; dims = (1, 3))[:],
        n_par.grid_k,
        n_par.transition_type,
    )

    # Aggregate savings in liquid assets
    supply_liquid = aggregate_asset_helper(
        sum(distr; dims = (2, 3))[:],
        n_par.grid_b,
        n_par.transition_type,
    )
    return supply_liquid, supply_illiquid
end

## ------------------------------------------------------------------------------------
## Other stuff
## ------------------------------------------------------------------------------------

"""
    get_asset_distr(distr, asset)

Extract the marginal distribution for a specific asset from various distribution types.

Arguments:

  - `distr::Union{CDF, CopulaOneAsset, CopulaTwoAssets, RepAgent}`: The distribution of
    agents
  - `asset::Symbol`: The asset to extract (`:b`, `:k`, or `:h`)

Returns:

  - Vector representing the marginal distribution for the specified asset.
"""
function get_asset_distr(distr::CDF, asset::Symbol)
    ndims_CDF = ndims(distr.CDF)
    if asset == :b && ndims_CDF == 2
        return distr.CDF[:, end]
    elseif asset == :b && ndims_CDF == 3
        return distr.CDF[:, end, end]
    elseif asset == :k && ndims_CDF == 3
        return distr.CDF[end, :, end]
    elseif asset == :h && ndims_CDF == 2
        return distr.CDF[end, :]
    else
        # asset == :h && ndims_CDF == 3
        return distr.CDF[end, end, :]
    end
end

function get_asset_distr(distr::Union{CopulaOneAsset,CopulaTwoAssets}, asset::Symbol)
    return getfield(distr, asset)
end

function get_asset_distr(distr::RepAgent, asset::Symbol)
    return [1.0]
end

"""
    eval_cdf(distr, asset, n_par, bound=nothing)

Evaluate the cumulative distribution function given either a PDF or CDF distribution object
for a specific asset.

Arguments:

  - `distr::DistributionValues` (or related Copula/CDF types): distribution of agents
  - `asset::Symbol`: asset to evaluate (`:b`, `:k`, or `:h`)
  - `n_par::NumericalParameters`: numerical parameters with grids
  - `bound`: optional upper bound; if `nothing` evaluates over full grid

Returns:

  - Value of the CDF up to `bound` (uses PDF->CDF conversion if input is a PDF)
"""
function eval_cdf(
    distr::Union{CopulaPDFsOneAsset,CopulaPDFsTwoAssets},
    asset::Symbol,
    n_par::NumericalParameters,
    bound = nothing,
)
    distr = get_asset_distr(distr, asset)
    grid = getfield(n_par, Symbol("grid_", asset))
    return aggregate_pdf(distr, grid, bound)
end

function eval_cdf(
    distr::Union{CDF,CopulaCDFsOneAsset,CopulaCDFsTwoAssets},
    asset::Symbol,
    n_par::NumericalParameters,
    bound = nothing,
)
    distr = get_asset_distr(distr, asset)
    grid = getfield(n_par, Symbol("grid_", asset))
    return integrate_cdf(distr, grid, bound)
end

function eval_cdf(
    distr::RepAgent,
    asset::Symbol,
    n_par::NumericalParameters,
    bound = nothing,
)
    return 1.0
end

aggregate_pdf(pdf::AbstractVector, grid::AbstractVector, bound::Nothing) = sum(pdf)
aggregate_pdf(pdf::AbstractVector, grid::AbstractVector, bound::Float64) =
    sum(pdf .* (grid .< bound))

function integrate_cdf(cdf::AbstractVector, grid::AbstractVector, bound::Nothing)
    grid_to_cdf_splines = Interpolator(grid, cdf)
    return integrate(grid_to_cdf_splines, grid[1], grid[end])
end

function integrate_cdf(cdf::AbstractVector, grid::AbstractVector, bound::Real)
    grid_to_cdf_splines = Interpolator(grid, cdf)
    return integrate(grid_to_cdf_splines, grid[1], bound)
end
