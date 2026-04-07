"""
    distrSummaries(distr, q, pf, n_par, net_income, gross_income, m_par)

Compute distributional summary statistics for a household distribution.

This file provides two dispatched implementations:

  - `distrSummaries(distr::RepAgent, ...)` — placeholder implementation that currently
    returns small numeric placeholders (`eps()`).
  - `distrSummaries(distr::Union{CDF,CopulaOneAsset,CopulaTwoAssets}, ...)` — concrete
    implementation that returns the tuple `(TOP10Wshare, TOP10Ishare, TOP10Inetshare, giniwealth, giniconsumption, sdlogy)`.

# Arguments

  - `distr`: distribution object (`RepAgent`, `CDF`, or `Copula*`).
  - `q::Real`: price of illiquid asset (used to compute total wealth when applicable).
  - `pf::PolicyFunctions`: policy functions (used to compute consumption summaries).
  - `n_par::NumericalParameters`: numerical/grid parameters and metadata.
  - `net_income`, `gross_income`: arrays following the codebase's internal indexing
    convention (see `distr_summaries_incomes` docstring for index usage).
  - `m_par::ModelParameters`: model parameters container.

# Returns

  - For the `CDF`/`Copula` dispatch, a 6-tuple:

      + `TOP10Wshare::Float64` — top 10% wealth share.
      + `TOP10Ishare::Float64` — top 10% gross income share.
      + `TOP10Inetshare::Float64` — top 10% net income share.
      + `giniwealth::Float64` — Gini coefficient of the wealth distribution.
      + `giniconsumption::Float64` — Gini coefficient of consumption.
      + `sdlogy::Float64` — standard deviation of log labor earnings.

Notes

  - The `RepAgent` dispatch is currently a stub and should be implemented if used.
"""
function distrSummaries(
    distr::RepAgent,
    q::Real,
    pf::PolicyFunctions,
    n_par,
    net_income::AbstractArray,
    gross_income::AbstractArray,
    m_par::ModelParameters,
)
    return eps(), eps(), eps(), eps(), eps(), eps()
end

function distrSummaries(
    distr::Union{CDF,CopulaOneAsset,CopulaTwoAssets},
    q::Real,
    pf::PolicyFunctions,
    n_par,
    net_income::AbstractArray,
    gross_income::AbstractArray,
    m_par::ModelParameters,
)
    TOP10Wshare, giniwealth = distr_summaries_wealth(distr, q, n_par)
    giniconsumption = distr_summaries_consumption(pf, net_income[5], distr, n_par)
    TOP10Ishare, TOP10Inetshare, sdlogy =
        distr_summaries_incomes(net_income, gross_income, distr, n_par)

    return TOP10Wshare, TOP10Ishare, TOP10Inetshare, giniwealth, giniconsumption, sdlogy
end

"""
    distr_summaries_wealth(distr, q, n_par)

Calculate wealth distribution statistics (TOP10 wealth share and Gini coefficient) using
multiple dispatch based on:
1. Distribution type: CDF vs Copula (CopulaOneAsset, CopulaTwoAssets)
2. Transition type: LinearTransition vs NonLinearTransition
3. Model type: OneAsset vs TwoAsset

Returns: (TOP10Wshare, giniwealth)
"""

function distr_summaries_wealth(
    distr::Union{CDF,CopulaOneAsset,CopulaTwoAssets},
    q::Real,
    n_par::NumericalParameters,
)

    # println("distr.COP: ", distr.COP)

    # Common setup for all CDF methods
    wealth_grid = total_wealth_grid(q, n_par, n_par.model)
    wealth_cdf = total_wealth_cdf(distr, n_par.model)

    IX = sortperm(wealth_grid)
    wealth_grid = wealth_grid[IX]
    wealth_cdf = wealth_cdf[IX]
    # @assert all(diff(wealth_cdf) .>= 0)
    # println("wealth_cdf after sorting: ", wealth_cdf)

    TOP10Wshare = topXshare(wealth_grid, wealth_cdf, 10.0, n_par.transition_type)
    giniwealth = gini(wealth_cdf, wealth_grid, n_par.transition_type)
    return TOP10Wshare, giniwealth
end

total_wealth_grid(q, n_par::NumericalParameters, model::OneAsset) = n_par.grid_b
total_wealth_grid(q, n_par::NumericalParameters, model::TwoAsset) =
    vec(n_par.grid_b' .+ q .* n_par.grid_k)
total_wealth_cdf(distr::CDF, model::OneAsset) = distr.CDF[:, end][:]
total_wealth_cdf(distr::CDF, model::TwoAsset) = distr.CDF[:, :, end][:]
total_wealth_cdf(distr::CopulaOneAsset, model::OneAsset) = distr.COP[:, end][:]
total_wealth_cdf(distr::CopulaTwoAssets, model::TwoAsset) = distr.COP[:, :, end][:]

"""
    find_wealth_at_percentile(wealth_grid, wealth_cdf, p, transition_type)

Invert a CDF to obtain the wealth level at cumulative probability `p ∈ [0,1]`.

Arguments

  - `wealth_grid`: grid of wealth values (ascending order recommended).
  - `wealth_cdf`: cumulative distribution values corresponding to `wealth_grid` (last element
    should be ≈ 1.0).
  - `p`: cumulative probability in [0,1] (use 0.9 for the 90th percentile).
  - `transition_type`: dispatches interpolation method; `NonLinearTransition` uses smooth
    splines, `LinearTransition` uses linear interpolation.

Returns

  - wealth level (scalar) at cumulative probability `p`.
"""
find_wealth_at_percentile(
    wealth_grid::AbstractVector,
    wealth_cdf::AbstractVector,
    percentile::Real,
    transition_type::NonLinearTransition,
) = Interpolator(wealth_cdf, wealth_grid)(percentile)

find_wealth_at_percentile(
    wealth_grid::AbstractVector,
    wealth_cdf::AbstractVector,
    percentile::Real,
    transition_type::LinearTransition,
) = mylinearinterpolate(wealth_cdf, wealth_grid, [percentile])[1]

"""
    topXshare(grid, cdf, X::Real, transition_type)

Compute the share of total wealth held by the top `X` percent of households.

Arguments

  - `grid`: wealth grid (values corresponding to `cdf`).
  - `cdf`: cumulative distribution function evaluated on `grid` (last element ≈ 1).
  - `X`: percentage in (0,100) representing top X percent (e.g., `10.0` for top 10%).
  - `transition_type`: controls interpolation/integration method (Linear vs NonLinear).

Notes

  - Internally the code converts the percentage `X` to a cumulative probability `p = 1 - X/100`.
  - For `NonLinearTransition` the code uses spline inversion and `aggregate_asset_helper` to
    integrate wealth above the threshold. For `LinearTransition` a PDF-based linear
    interpolation is used. These two approaches can yield small numerical differences.
"""
function topXshare(grid, cdf, X::Real, transition_type::NonLinearTransition)
    @assert 0.0 < X < 100.0 "X must be between 0 and 100"
    wealth_at_X = find_wealth_at_percentile(grid, cdf, 1.0 - X / 100.0, transition_type)
    total = aggregate_asset_helper(cdf, grid, transition_type, nothing, false)
    total_up_to_X = aggregate_asset_helper(cdf, grid, transition_type, wealth_at_X, false)
    return 1.0 - total_up_to_X / total
end

function topXshare(grid, cdf, X::Real, transition_type::LinearTransition)
    @assert 0.0 < X < 100.0 "X must be between 0 and 100"
    pdf = cdf_to_pdf(cdf)
    cumulative = cumsum(grid .* pdf)
    shares = cumulative ./ cumulative[end]
    return 1.0 - mylinearinterpolate(cdf, shares, [1.0 - X / 100.0])[1]
end

"""
    distr_summaries_consumption(pf, aux_x, distr, n_par)

Compute the Gini coefficient for consumption.

Returns a scalar Gini coefficient computed from consumption values and the distribution
induced by `distr` and the policies `pf`.

Note: Because some CDF representations can contain flat segments (zero increases) we remove
zero-increment points before calling the Gini routines to ensure monotonicity required by
spline-based methods.
"""
function distr_summaries_consumption(
    pf::PolicyFunctions,
    aux_x::AbstractArray,
    distr::Union{CDF,CopulaOneAsset,CopulaTwoAssets},
    n_par::NumericalParameters,
)
    c, cdf_c = get_distr_c_sorted(pf, aux_x, distr, n_par)
    # Remove zero increases from cdf_c and according c values because not strictly increasing CDF gave problems with non-linear gini
    idxs = findall(diff(cdf_c) .== 0.0)
    c = deleteat!(c, idxs .+ 1)
    cdf_c = deleteat!(cdf_c, idxs .+ 1)
    giniconsumption = gini(cdf_c, c, n_par.transition_type)
    return giniconsumption
end

"""
    get_distr_c_sorted(pf::PolicyFunctionsOneAsset, aux_x, distr, n_par)

Return sorted consumption values and the corresponding cumulative distribution induced by
`distr`.

Outputs are 1D arrays: `(c_sorted, cdf_c)` suitable for computing Gini.
"""
function get_distr_c_sorted(
    pf::PolicyFunctionsOneAsset,
    aux_x,
    distr,
    n_par::NumericalParameters,
)
    c = pf.x_n_star .+ aux_x
    # TODO - Adapt for correct CDF treatment (do not use pdfs or correct ones (derivative of CDFs))
    IX = sortperm(c[:])
    c[:] .= c[IX]
    distr_c = cdf_to_pdf(get_joint_CDF(distr))[IX]
    distr_c = cumsum(distr_c)
    return c[:], distr_c[:]
end

"""
    get_distr_c_sorted(pf::PolicyFunctionsTwoAssets, aux_x, distr, n_par)

Like the one-asset variant, but returns consumption values for both adjustment states
concatenated and the corresponding cumulative distribution.
"""
function get_distr_c_sorted(
    pf::PolicyFunctionsTwoAssets,
    aux_x,
    distr,
    n_par::NumericalParameters,
)
    c = Array{eltype(pf.x_a_star)}(undef, (n_par.nb, n_par.nk, n_par.nh, 2))
    distr_c = similar(c)
    c[:, :, :, 1] .= pf.x_a_star .+ aux_x
    c[:, :, :, 2] .= pf.x_n_star .+ aux_x
    distr_c[:, :, :, 1] .= n_par.m_par.λ .* get_joint_CDF(distr)
    distr_c[:, :, :, 2] .= (1 - n_par.m_par.λ) .* get_joint_CDF(distr)

    IX = sortperm(c[:])
    c[:] .= c[IX]
    distr_c[:] .= distr_c[IX]

    return c[:], distr_c[:]
end

"""
    distr_summaries_incomes(net_income, gross_income, distr, n_par)

Compute top-10 gross and net income shares and the standard deviation of log labor earnings.
For the income definitions, see the indexing conventions in the codebase [`incomes`](@ref).
"""
function distr_summaries_incomes(
    net_income::AbstractArray,
    gross_income::AbstractArray,
    distr::Union{CDF,CopulaOneAsset,CopulaTwoAssets},
    n_par::NumericalParameters,
)
    Y_pdf = cdf_to_pdf(get_joint_CDF(distr))
    capital_inc = net_income[2] .+ net_income[3] .- n_par.mesh_b
    Yidio = net_income[6] .+ capital_inc
    IX = sortperm(Yidio[:])
    Yidio = Yidio[IX]
    Y_cdf = cumsum(Y_pdf[IX])
    Y_w = Yidio .* Y_pdf[IX]
    net_incomeshares = cumsum(Y_w) ./ sum(Y_w)
    TOP10Inetshare = 1.0 .- mylinearinterpolate(Y_cdf, net_incomeshares, [0.9])[1]

    # Top 10 gross income share
    Yidio = gross_income[1] .+ capital_inc
    IX = sortperm(Yidio[:])
    Yidio = Yidio[IX]
    Y_cdf = cumsum(Y_pdf[IX])
    Y_w = Yidio .* Y_pdf[IX]
    incomeshares = cumsum(Y_w) ./ sum(Y_w)
    TOP10Ishare = 1.0 .- mylinearinterpolate(Y_cdf, incomeshares, [0.9])[1]

    sdlogy = get_sdlogy(distr, gross_income, n_par)
    return TOP10Ishare, TOP10Inetshare, sdlogy
end

"""
    get_all_but_last_dim(a)

Return all dimensions of `a` except the last one (useful when the last dim indexes discrete
states such as a binary adjustment indicator).
"""
function get_all_but_last_dim(a)
    last_dim = ndims(a)
    return selectdim(a, last_dim, 1:(size(a, last_dim) - 1))
end

"""
    get_sdlogy(distr, gross_income, n_par)

Compute the standard deviation of (log) labor earnings using the joint distribution `distr`
and the provided `gross_income` array.

The function expects `gross_income[1]` to contain the labor earnings grid used in the joint
distribution; the routine flattens non-last dimensions and uses the joint PDF normalized to
sum to 1.
"""
function get_sdlogy(distr, gross_income, n_par)

    # get all, but last dimension
    Yidio = get_all_but_last_dim(gross_income[1])
    distr_aux = get_all_but_last_dim(cdf_to_pdf(get_joint_CDF(distr)))

    IX = sortperm(Yidio[:])
    Yidio = Yidio[IX]
    distr_aux = distr_aux ./ sum(distr_aux[:])
    Y_pdf = distr_aux[IX]

    return sqrt(dot(Y_pdf, Yidio .^ 2) .- dot(Y_pdf, Yidio) .^ 2)
end

"""
    gini(x, pdf)

Compute the Gini coefficient for a discrete distribution.

Preconditions

  - `x` should be sorted in ascending order and `pdf` should represent non-negative weights
    that sum to 1 (or be proportional to probabilities; the routine uses cumulative sums
    internally).

# Arguments

  - `x::AbstractArray`: Values (e.g., income or wealth) in ascending order.
  - `pdf::AbstractArray`: Corresponding probability weights or PDF values.

# Returns

  - `gini::Float64`: Gini coefficient in [0,1].
"""
function gini(x, pdf)
    s = 0.0
    gini = 0.0
    for i in eachindex(x)
        gini -= pdf[i] * s
        s += x[i] * pdf[i]
        gini -= pdf[i] * s
    end
    gini /= s
    gini += 1.0
    return gini
end

"""
gini_by_integration(grid::AbstractArray, cdf::AbstractArray, μ::Number)

Compute the Gini coefficient via numerical integration using a monotone spline of the CDF.
Implements the identity

    G = 1 - (1/μ) ∫ (1 - F(x))^2 dx,

where `F(x)` is the CDF and `μ` is the mean of the distribution. The CDF should be evaluated
on `grid` and the last CDF element should be ≈ 1.0.
"""
function gini_by_integration(grid::AbstractArray, cdf::AbstractArray, μ::Number)
    # @assert cdf[end] ≈ 1.0 "The last element of the CDF must be 1.0"
    itp_one_minus_CDF_squared = Interpolator(grid, (1.0 .- cdf) .^ 2)
    return 1.0 .- (1.0 ./ μ) .* integrate(itp_one_minus_CDF_squared, grid[1], grid[end])
end

gini(cdf, grid, transition_type::LinearTransition) = gini(grid, cdf_to_pdf(cdf))
gini(cdf, grid, transition_type::NonLinearTransition) = gini_by_integration(
    grid,
    cdf,
    aggregate_asset_helper(cdf, grid, transition_type, nothing, false),
)
