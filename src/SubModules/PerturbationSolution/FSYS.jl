"""
    Fsys(X, XPrime, XSS, m_par, n_par, indexes, compressionIndexes, transform_elements; only_F = true)

Equilibrium error function: returns deviations from equilibrium around steady state.

It splits the computation into an *Aggregate Part*, handled by [`Fsys_agg()`](@ref), and a *Heterogeneous Agent Part* (backward iteration of value functions and forward iteration of distributions).

# Arguments

  - `X`,`XPrime`: Deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
  - `XSS`: States and controls in steady state
  - `m_par`: Model parameters
  - `n_par`: Numerical parameters
  - `indexes`: Index struct to access variables in `X` and `XPrime`
  - `compressionIndexes`: Indices of DCT coefficients selected for perturbation
  - `transform_elements`: A struct containing transformation matrices (`Γ`, `DC`, `IDC`, `DCD`, `IDCD`) used to map between compressed states/controls and full grids (distributions and value functions).
  - `only_F::Bool`: If `true` (default), returns only the error vector `F`. If `false`, returns additional objects useful for debugging or other computations.

# Returns

  - If `only_F = true`:

      + `F`: Vector of equilibrium errors.

  - If `only_F = false`:

      + `F`: Vector of equilibrium errors.
      + `pf`: Policy functions for the current iteration.
      + `vf_new`: Updated value functions.
      + `tax_rev`: Tax revenue.
"""
function Fsys(
    X::AbstractArray,
    XPrime::AbstractArray,
    XSS::Array{Float64,1},
    m_par::ModelParameters,
    n_par::NumericalParameters,
    indexes::IndexStruct,
    compressionIndexes::Vector,
    transform_elements::Transformations,
    only_F = true,
)

    ## ------------------------------------------------------------------------------------
    ## Preamble
    ## ------------------------------------------------------------------------------------

    # Initialize the output vector, use the same type as the input
    F = zeros(eltype(X), size(X))

    ## Unpack aggregate variables -------------------------------------------------------

    # Unpack X, XPrime, and XSS into variables
    @generate_equations()

    ## Unpack perturbed distributions -----------------------------------------------------

    distrSS = unpack_ss_distributions(XSS, indexes.distrSS, n_par)
    distr, distrPrime = unpack_perturbed_distributions(
        X,
        XPrime,
        distrSS,
        indexes.distr,
        compressionIndexes,
        transform_elements,
        n_par,
    )

    ## Unpack value functions -------------------------------------------------------------

    log_inv_vfSS = unpack_ss_valuefunctions(XSS, indexes.valueFunctionSS, m_par, n_par)
    vfPrime = unpack_perturbed_valuefunctions(
        XPrime,
        indexes.valueFunction,
        log_inv_vfSS,
        transform_elements,
        compressionIndexes,
        n_par,
    )

    ## Unpack transition matrix -----------------------------------------------------------

    Π = unpack_transition_matrix(X, σ, n_par, m_par, n_par.model)

    ## ------------------------------------------------------------------------------------
    ## Equilibrium conditions (aggregate)
    ## ------------------------------------------------------------------------------------

    ## Aggregate equations ----------------------------------------------------------------
    # Load aggregate equations as specified in the model file
    F = Fsys_agg(X, XPrime, XSS, distr, m_par, n_par, indexes)

    ## Update distributional statistics ---------------------------------------------------
    # Update distributional statistics based on heterogeneous agent part

    # Scaling factor for individual productivity
    F[indexes.Htilde] =
        (log(Htilde)) - (log(
            dot(
                get_slice_h(get_PDF_h(distr), Val(n_par.entrepreneur)),
                get_slice_h(n_par.grid_h, Val(n_par.entrepreneur)),
            ),
        ))

    # Asset market clearing conditions
    error_term_assets!(F, distr, n_par, indexes, n_par.model, @asset_vars(n_par.model)...)

    ## ------------------------------------------------------------------------------------
    ## Equilibrium conditions (idiosyncratic)
    ## ------------------------------------------------------------------------------------

    ## Incomes ----------------------------------------------------------------------------
    # Calculate incomes based on the model-specific income functions

    @write_args_hh_prob()

    # Calculate net income and effective interest rate
    net_income, gross_income, eff_int = incomes(n_par, m_par, args_hh_prob)

    ## Policy and value functions ---------------------------------------------------------

    # Calculate expected marginal value functions
    beta_factor = "beta" in state_names ? beta : 1.0
    EvfPrime = expected_marginal_values(Π, vfPrime, n_par, beta_factor)

    # Calculate policy functions (policy iteration)
    pf = EGM_policyupdate(
        EvfPrime,
        args_hh_prob,
        net_income,
        n_par,
        m_par,
        false,
        n_par.model,
    )

    # Update marginal values (marginal utilities and logs)
    vf_err = updateW(EvfPrime, pf, args_hh_prob, m_par, n_par, beta_factor)

    vf_err.b .*= eff_int
    if !only_F
        vf_new = copy(vf_err)
    end

    # Update distribution via direct transition
    distrPrimeUpdate = DirectTransition(pf, distr, m_par.λ, Π, n_par)

    ## Set up the error terms for idiosyncratic part --------------------------------------

    # Error terms on marginal values (controls)
    error_term_vf!(
        X,
        F,
        vf_err,
        log_inv_vfSS,
        indexes.valueFunction,
        transform_elements,
        compressionIndexes,
        n_par,
    )

    # Error terms on marginal distributions (levels, state deviations)
    error_term_distr!(
        F,
        distrPrimeUpdate,
        distrPrime,
        distr,
        distrSS,
        Π,
        indexes.distr,
        transform_elements,
        compressionIndexes,
        n_par,
        n_par.transf_CDF,
    )

    ## ------------------------------------------------------------------------------------
    ## Equilibrium conditions (auxiliary statistics)
    ## ------------------------------------------------------------------------------------

    # Calculate distribution statistics (generalized moments)
    TOP10WshareT, TOP10IshareT, TOP10InetshareT, GiniWT, GiniCT, sdlogyT =
        distrSummaries(distr, q, pf, n_par, net_income, gross_income, m_par)

    # Error terms on distribution summaries
    F[indexes.GiniW] = log.(GiniW) - log.(GiniWT)
    F[indexes.TOP10Ishare] = log.(TOP10Ishare) - log.(TOP10IshareT)
    F[indexes.TOP10Inetshare] = log.(TOP10Inetshare) - log.(TOP10InetshareT)
    F[indexes.TOP10Wshare] = log.(TOP10Wshare) - log.(TOP10WshareT)
    F[indexes.GiniC] = log.(GiniC) - log.(GiniCT)
    F[indexes.sdlogy] = log.(sdlogy) - log.(sdlogyT)

    ## ------------------------------------------------------------------------------------
    ## Return
    ## ------------------------------------------------------------------------------------

    if only_F
        return F
    else
        return F, pf, vf_new, ((Tbar .- 1.0) * (wH * N) + (Tbar .- 1.0) * Π_E)
    end
end

"""
error_term_vf!(X, F, vf_updated, log_inv_vfSS, indexes, compres, compressionIndexes, n_par)

Populate equilibrium residuals for marginal value functions (`vf`) across model variants.

Applies to methods for:

  - `ValueFunctionsTwoAssets` → writes `F[indexes.b]` and `F[indexes.k]`.
  - `ValueFunctionsOneAsset` → writes `F[indexes.b]`.
  - `ValueFunctionsCompleteMarkets` → no-op.

Behavior:

  - Converts updated marginal utilities to log deviations from steady state (`log_inv_vfSS`).
  - Compresses deviations via DCT (`compres.DC`, `compres.IDC`) using `compressionIndexes[1]`.

Arguments:

  - `X`, `F`: current deviation vector and residual vector.
  - `vf_updated`, `log_inv_vfSS`: updated and steady-state marginal values.
  - `indexes`: value-function index ranges for writing into `F`.
  - `compres`: transformation elements (DCT forward/inverse).
  - `compressionIndexes`: coefficient selections for compression.
  - `n_par`: numerical parameters.

Notes:

  - Shapes of `vf_updated.*` must match grids; `log_inv_vfSS.*` are broadcast reshaped.
"""
function error_term_vf!(
    X::AbstractVector,
    F::AbstractVector,
    vf_updated::ValueFunctionsTwoAssets,
    log_inv_vfSS::ValueFunctionsTwoAssets,
    indexes::ValueFunctionsTwoAssetsIndexes,
    compres::Transformations,
    compressionIndexes::AbstractVector,
    n_par::NumericalParameters,
)
    invmutil!(vf_updated.b, vf_updated.b, n_par.m_par)
    invmutil!(vf_updated.k, vf_updated.k, n_par.m_par)
    n_dims = size(vf_updated.b)
    vf_updated.b .= log.(vf_updated.b) .- reshape(log_inv_vfSS.b, n_dims)
    vf_updated.k .= log.(vf_updated.k) .- reshape(log_inv_vfSS.k, n_dims)
    vb_thet = compress(
        compressionIndexes[1][1],
        vf_updated.b,
        compres.DC,
        compres.IDC,
        n_par.model,
    )
    vk_thet = compress(
        compressionIndexes[1][2],
        vf_updated.k,
        compres.DC,
        compres.IDC,
        n_par.model,
    )
    F[indexes.b] = X[indexes.b] .- vb_thet
    F[indexes.k] = X[indexes.k] .- vk_thet
end

function error_term_vf!(
    X,
    F,
    vf_updated::ValueFunctionsOneAsset,
    log_inv_vfSS::ValueFunctionsOneAsset,
    indexes::ValueFunctionsOneAssetIndexes,
    compres::Transformations,
    compressionIndexes::AbstractVector,
    n_par::NumericalParameters,
)
    invmutil!(vf_updated.b, vf_updated.b, n_par.m_par)
    vf_updated.b .= log.(vf_updated.b) .- reshape(log_inv_vfSS.b, size(vf_updated.b))
    v_thet = compress(
        compressionIndexes[1][1],
        vf_updated.b,
        compres.DC,
        compres.IDC,
        n_par.model,
    )
    F[indexes.b] = X[indexes.b] .- v_thet
end

function error_term_vf!(
    X,
    F,
    vf_updated::ValueFunctionsCompleteMarkets,
    log_inv_vfSS::ValueFunctionsCompleteMarkets,
    indexes::ValueFunctionsCompleteMarketsIndexes,
    compres::Transformations,
    compressionIndexes::AbstractVector,
    n_par::NumericalParameters,
)
    # No value function errors for complete markets
end

"""
error_term_distr!(F, distrPrimeUpdate, distrPrime, distr, distrSS, Π, indexes_distr, compres, compressionIndexes, n_par)

Populate residuals for marginal distributions and copulas across distribution types.

Applies to methods for:

  - `CopulaTwoAssets` and `CopulaOneAsset`: writes residuals for asset/labor marginals and copula coefficients.
  - `CDF`: representative-agent CDF residuals.
  - `RepAgent`: representative-agent labor residuals.

Behavior:

  - Marginals: level differences between updated (`distrPrimeUpdate`) and target (`distrPrime`).
  - Copula: deviation of iterated copula from steady-state copula, compressed via DCT (`compres.DCD`, `compres.IDCD`).

Arguments:

  - `F`: residual vector to write into index ranges.
  - `Π`: transition matrix used for labor transitions when applicable.
  - `indexes_distr`: indexes for each distribution component.
  - `compressionIndexes[2]`: coefficient selection for copula compression (when applicable).
"""
function error_term_distr!(
    F,
    distrPrimeUpdate::CopulaTwoAssets,
    distrPrime::CopulaTwoAssets,
    distr::CopulaTwoAssets,
    distrSS::CopulaTwoAssets,
    Π,
    indexes_distr::CopulaTwoAssetsIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::LinearTransformation,
)
    dt = typeof(distrSS) # distribution type

    distr_hPrimeUpdate = (get_PDF_h(distr)' * Π)[:]
    # Error Terms on marginal distribution (in levels, states)
    F[indexes_distr.b] = (distrPrimeUpdate.b .- distrPrime.b)[1:(end - 1)]
    F[indexes_distr.k] = (distrPrimeUpdate.k .- distrPrime.k)[1:(end - 1)]
    F[indexes_distr.h] = (distr_hPrimeUpdate .- get_PDF_h(distrPrime))[1:(end - 1)]

    # Error Terms on copula (states): deviation of iterated copula from fixed copula
    CDF_b_PrimeUp = get_CDF(distrPrimeUpdate.b, dt)
    CDF_k_PrimeUp = get_CDF(distrPrimeUpdate.k, dt)
    CDF_h_PrimeUp = get_CDF(distrPrimeUpdate.h, dt)
    # steady state copula marginals (cdfs)
    s_m_b = n_par.copula_marginal_b .+ zeros(eltype(F), 1)
    s_m_k = n_par.copula_marginal_k .+ zeros(eltype(F), 1)
    s_m_h = n_par.copula_marginal_h .+ zeros(eltype(F), 1)
    CopulaDevPrime(x::Vector, y::Vector, z::Vector) =
        myinterpolate3(
            CDF_b_PrimeUp,
            CDF_k_PrimeUp,
            CDF_h_PrimeUp,
            distrPrimeUpdate.COP,
            n_par.model,
            x,
            y,
            z,
        ) .- myinterpolate3(
            get_CDF(distrSS.b, dt),
            get_CDF(distrSS.k, dt),
            get_CDF(distrSS.h, dt),
            distrSS.COP,
            n_par.model,
            x,
            y,
            z,
        )
    CDF_Dev = CopulaDevPrime(s_m_b, s_m_k, s_m_h)
    COP_thet = compress(
        compressionIndexes[2],
        cdf_to_pdf(CDF_Dev - distrPrime.COP),
        compres.DCD,
        compres.IDCD,
        n_par.model,
    )
    F[indexes_distr.COP] = COP_thet
end

function error_term_distr!(
    F,
    distrPrimeUpdate::CopulaTwoAssets,
    distrPrime::CopulaTwoAssets,
    distr::CopulaTwoAssets,
    distrSS::CopulaTwoAssets,
    Π,
    indexes_distr::CopulaTwoAssetsIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::ParetoTransformation,
)
    dt = typeof(distrSS) # distribution type

    distr_hPrimeUpdate = (get_PDF_h(distr)' * Π)[:]
    # Error Terms on marginal distribution (in levels, states)
    F[indexes_distr.b] = (distrPrimeUpdate.b .- distrPrime.b)[1:(end - 1)]
    F[indexes_distr.k] = (distrPrimeUpdate.k .- distrPrime.k)[1:(end - 1)]
    F[indexes_distr.h] = (distr_hPrimeUpdate .- get_PDF_h(distrPrime))[1:(end - 1)]

    # Error Terms on copula (states): deviation of iterated copula from fixed copula
    CDF_b_PrimeUp = get_CDF(distrPrimeUpdate.b, dt)
    CDF_k_PrimeUp = get_CDF(distrPrimeUpdate.k, dt)
    CDF_h_PrimeUp = get_CDF(distrPrimeUpdate.h, dt)
    # steady state copula marginals (cdfs)
    s_m_b = n_par.copula_marginal_b .+ zeros(eltype(F), 1)
    s_m_k = n_par.copula_marginal_k .+ zeros(eltype(F), 1)
    s_m_h = n_par.copula_marginal_h .+ zeros(eltype(F), 1)
    CopulaDevPrime(x::Vector, y::Vector, z::Vector) =
        myinterpolate3(
            CDF_b_PrimeUp,
            CDF_k_PrimeUp,
            CDF_h_PrimeUp,
            distrPrimeUpdate.COP,
            n_par.model,
            x,
            y,
            z,
        ) .- myinterpolate3(
            get_CDF(distrSS.b, dt),
            get_CDF(distrSS.k, dt),
            get_CDF(distrSS.h, dt),
            distrSS.COP,
            n_par.model,
            x,
            y,
            z,
        )
    CDF_Dev = CopulaDevPrime(s_m_b, s_m_k, s_m_h)
    COP_thet = compress(
        compressionIndexes[2],
        cdf_to_pdf(CDF_Dev - distrPrime.COP),
        compres.DCD,
        compres.IDCD,
        n_par.model,
    )
    F[indexes_distr.COP] = COP_thet
end

function error_term_distr!(
    F,
    distrPrimeUpdate::CopulaOneAsset,
    distrPrime::CopulaOneAsset,
    distr::CopulaOneAsset,
    distrSS::CopulaOneAsset,
    Π,
    indexes_distr::CopulaOneAssetIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::LinearTransformation,
)
    tt = n_par.transition_type

    distr_hPrimeUpdate = (get_PDF_h(distr)' * Π)[:]
    # Error Terms on marginal distribution (in levels, states)
    F[indexes_distr.b] = (distrPrimeUpdate.b .- distrPrime.b)[1:(end - 1)]
    F[indexes_distr.h] = (distr_hPrimeUpdate .- get_PDF_h(distrPrime))[1:(end - 1)]

    # Steady state copula marginals (cdfs)
    s_m_b = n_par.copula_marginal_b .+ zeros(eltype(F), 1)
    s_m_h = n_par.copula_marginal_h .+ zeros(eltype(F), 1)
    # Error Terms on copula (states): deviation of iterated copula from fixed copula
    CDF_Dev = CopulaDevPrime(s_m_b, s_m_h, distrSS, distrPrimeUpdate, n_par, tt)
    COP_thet = compress(
        compressionIndexes[2],
        cdf_to_pdf(CDF_Dev - distrPrime.COP),
        compres.DCD,
        compres.IDCD,
        n_par.model,
    )
    F[indexes_distr.COP] = COP_thet
end

function error_term_distr!(
    F,
    distrPrimeUpdate::CopulaOneAsset,
    distrPrime::CopulaOneAsset,
    distr::CopulaOneAsset,
    distrSS::CopulaOneAsset,
    Π,
    indexes_distr::CopulaOneAssetIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::ParetoTransformation,
)
    # make additional check here: can't use pareto transformation with PDF representation
    if isa(distrSS, CopulaPDFsOneAsset)
        return error_term_distr!(
            F,
            distrPrimeUpdate,
            distrPrime,
            distr,
            distrSS,
            Π,
            indexes_distr,
            compres,
            compressionIndexes,
            n_par,
            LinearTransformation(),
        )
    end
    tt = n_par.transition_type

    distr_hPrimeUpdate = (get_PDF_h(distr)' * Π)[:]
    # Error Terms on marginal distribution (in levels, states)
    Dev_CDF_m = zeros(eltype(distrPrime.b), n_par.nb - 1)
    Dev_CDF_m[1] = log.(distrPrimeUpdate.b[1])
    Dev_CDF_m[2] =
        log.(-log.((1.0 .- distrPrimeUpdate.b[2]) / (1.0 .- distrPrimeUpdate.b[1])))
    for i = 3:(n_par.nb - 1)
        Dev_CDF_m[i] =
            log.(
                -log.((1.0 .- distrPrimeUpdate.b[i]) / (1.0 .- distrPrimeUpdate.b[i - 1])) /
                log.(n_par.grid_b[i] / n_par.grid_b[i - 1])
            )
    end
    CDF_m_dev = Dev_CDF_m .- distrPrime.b

    F[indexes_distr.b] = CDF_m_dev
    F[indexes_distr.h] = (distr_hPrimeUpdate .- get_PDF_h(distrPrime))[1:(end - 1)]

    # Steady state copula marginals (cdfs)
    s_m_b = n_par.copula_marginal_b .+ zeros(eltype(F), 1)
    s_m_h = n_par.copula_marginal_h .+ zeros(eltype(F), 1)
    # Error Terms on copula (states): deviation of iterated copula from fixed copula
    CDF_Dev = CopulaDevPrime(s_m_b, s_m_h, distrSS, distrPrimeUpdate, n_par, tt)
    COP_thet = compress(
        compressionIndexes[2],
        cdf_to_pdf(CDF_Dev - distrPrime.COP),
        compres.DCD,
        compres.IDCD,
        n_par.model,
    )
    F[indexes_distr.COP] = COP_thet
end

function error_term_distr!(
    F,
    CDFsPrimeUpdate::CDF,
    CDFsPrime::CDF,
    CDFs::CDF,
    CDFsSS::CDF,
    Π,
    indexes_distr::CDFIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::LinearTransformation,
)
    F[indexes_distr.CDF] = (CDFsPrimeUpdate.CDF[:] .- CDFsPrime.CDF[:])[1:(end - 1)]
end

function error_term_distr!(
    F,
    CDFsPrimeUpdate::CDF,
    CDFsPrime::CDF,
    CDFs::CDF,
    CDFsSS::CDF,
    Π,
    indexes_distr::CDFIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::ParetoTransformation,
)
    Dev_CDF = zeros(eltype(CDFsPrime.CDF), n_par.nb, n_par.nh)
    CDF_jointSS_cond = copy(CDFsSS.CDF)
    CDF_jointSS_cond[:, 2:end] .= diff(CDF_jointSS_cond; dims = 2)
    PDF_ySS = CDF_jointSS_cond[end, :][:]
    CDF_jointPrimeUp = copy(CDFsPrimeUpdate.CDF)
    CDF_jointPrimeUp[:, 2:end] .= diff(CDF_jointPrimeUp; dims = 2)
    for i_h = 1:(n_par.nh)
        idx_start_pareto = compres.pareto_indices[1][i_h] + 1
        Dev_CDF[1:idx_start_pareto, i_h] = CDF_jointPrimeUp[1:idx_start_pareto, i_h]
        idx_end_pareto = compres.pareto_indices[2][i_h]
        for i = (idx_start_pareto + 1):(idx_end_pareto - 1)
            alp =
                -log.(
                    (PDF_ySS[i_h] .- CDF_jointPrimeUp[i, i_h]) ./
                    (PDF_ySS[i_h] .- CDF_jointPrimeUp[i - 1, i_h])
                ) ./ log.(n_par.grid_b[i] / n_par.grid_b[i - 1])
            (alp == 0.0) &&
                (@warn "alp = $alp, so log(alp) is -Inf for i_b = $i and i_h = $i_h")
            Dev_CDF[i, i_h] = log.(alp)
        end
        Dev_CDF[idx_end_pareto:end, i_h] = CDF_jointPrimeUp[idx_end_pareto:end, i_h]
    end
    CDF_dev = (Dev_CDF[:] .- CDFsPrime.CDF[:])[1:(end - 1)]
    F[indexes_distr.CDF] = CDF_dev
end

function error_term_distr!(
    F,
    CDFsPrimeUpdate::RepAgent,
    CDFsPrime::RepAgent,
    CDFs::RepAgent,
    CDFsSS::RepAgent,
    Π,
    indexes_distr::RepAgentIndexes,
    compres::Transformations,
    compressionIndexes::Vector,
    n_par::NumericalParameters,
    transftype::Union{LinearTransformation,ParetoTransformation},
)
    distr_hPrimeUpdate = (get_PDF_h(CDFs)' * Π)[:]
    # Error Terms on marginal distribution (in levels, states)
    F[indexes_distr.h] = (distr_hPrimeUpdate .- get_PDF_h(CDFsPrime))[1:(end - 1)]
end

"""
CopulaDevPrime(x, z, distrSS, distrPrimeUpdate, n_par, tt)

Evaluate copula CDF deviations given steady-state and updated marginals/copula.

Applies to methods for:

  - `CopulaPDFsOneAsset` with `tt::LinearTransition`: uses linear interpolation on CDFs.
  - `CopulaCDFsOneAsset` with `tt::NonLinearTransition`: uses `marginal_to_joint` to evaluate deviations on steady-state support.
"""
function CopulaDevPrime(
    x::AbstractArray,
    z::AbstractArray,
    distrSS::CopulaPDFsOneAsset,
    distrPrimeUpdate::CopulaPDFsOneAsset,
    n_par::NumericalParameters,
    tt::LinearTransition,
)
    dt = typeof(distrSS) # distribution type
    return mylinearinterpolate2(
        get_CDF(distrPrimeUpdate.b, dt),
        get_CDF(distrPrimeUpdate.h, dt),
        distrPrimeUpdate.COP,
        x,
        z,
    ) .- mylinearinterpolate2(
        get_CDF(distrSS.b, dt),
        get_CDF(distrSS.h, dt),
        distrSS.COP,
        x,
        z,
    )
end

function CopulaDevPrime(
    x::AbstractArray,
    z::AbstractArray,
    distrSS::CopulaCDFsOneAsset,
    distrPrimeUpdate::CopulaCDFsOneAsset,
    n_par::NumericalParameters,
    tt::NonLinearTransition,
)
    COP_Prime_at_ss = zeros(eltype(distrPrimeUpdate.COP), (size(distrPrimeUpdate.COP)))
    CDF_Dev = zeros(eltype(distrPrimeUpdate.COP), (n_par.nb_copula, n_par.nh_copula))
    COP_Prime_at_ss .=
        marginal_to_joint(distrPrimeUpdate.b, distrPrimeUpdate.COP, distrSS.b)
    CDF_Dev .= marginal_to_joint(
        distrSS.b,
        COP_Prime_at_ss .- distrSS.COP,
        x;
        monotonic_spline = false,
    )
    return CDF_Dev
end

"""
error_term_assets!(F, distr, n_par, indexes, model, args...)

Market-clearing residuals across asset structures:

  - `::OneAsset`: aggregate bonds equal household `:b` holdings; `BD` equals negative net bond position.
  - `::TwoAsset`: `TotalAssets = B + q*K`, and capital/bond markets clear at aggregates.
  - `::CompleteMarkets`: no residuals (no-op).

Arguments:

  - `distr`: distribution values to aggregate.
  - `indexes`: index ranges for writing residuals.
  - `args...`: additional variables per method (e.g., `TotalAssets`, `BD`, `K`, `B`, `q`).
"""
function error_term_assets!(
    F,
    distr::DistributionValues,
    n_par::NumericalParameters,
    indexes::IndexStruct,
    model::OneAsset,
    TotalAssets,
    BD,
)
    F[indexes.TotalAssets] = (log(TotalAssets)) - log(aggregate_asset(distr, :b, n_par))
    # IOUs
    F[indexes.BD] = (log(BD)) - (log(max(eps(), -aggregate_asset(distr, :b, n_par, 0.0)))) # ensure non-zero debt
end

function error_term_assets!(
    F,
    distr::DistributionValues,
    n_par::NumericalParameters,
    indexes::IndexStruct,
    model::TwoAsset,
    TotalAssets,
    BD,
    K,
    B,
    q,
)
    B_agg = aggregate_asset(distr, :b, n_par)
    K_agg = aggregate_asset(distr, :k, n_par)

    F[indexes.TotalAssets] = (log(TotalAssets)) - log(B_agg + q * K_agg)
    # IOUs
    F[indexes.BD] = (log(BD)) - (log(-aggregate_asset(distr, :b, n_par, 0.0)))

    # Capital market clearing
    F[indexes.K] = (log(K)) - (log(K_agg))

    # Bond market clearing
    F[indexes.B] = (log(B)) - (log(B_agg))
end

function error_term_assets!(
    F,
    distr::DistributionValues,
    n_par::NumericalParameters,
    indexes::IndexStruct,
    model::CompleteMarkets,
) end
