"""
    first_stage_reduction(
        vfSS,
        transition_matricesSS,
        pfSS,
        args_hh_prob,
        include_list_idx,
        n_par,
        m_par,
    )

First-stage reduction for selecting DCT (Discrete Cosine Transform) coefficients to perturb,
following Appendix C of BBL. Uses the steady-state value functions `vfSS`, policy functions
`pfSS`, and transition matrices to compute the Jacobian of marginal values with respect to
selected aggregate inputs, transform these via the inverse marginal utility, and select a
small set of DCT coefficients that best represent the response shapes.

  - `include_list_idx` selects which elements of `args_hh_prob` are perturbed; others remain
    at steady state.
  - Works for one-asset and two-asset models via multiple dispatch on `vfSS` and
    `transition_matricesSS`.
  - Complete-markets variant is a no-op selection (representative-agent shortcut) and returns
    an empty selection.

Returns a pair `(ind, J)` where `ind` contains selected DCT indices by asset
(liquid/illiquid when applicable) and `J` is the Jacobian of marginal values with respect to
the selected inputs.

# Arguments

  - `vfSS::ValueFunctions`: steady-state value functions.
  - `transition_matricesSS::TransitionMatrices`: steady-state exogenous transition matrices.
  - `pfSS::PolicyFunctions`: steady-state policy functions.
  - `args_hh_prob::AbstractVector{Float64}`: vector of household problem arguments at
    steady state (in original scale).
  - `include_list_idx::AbstractVector{Int}`: indexes of `args_hh_prob` to include in the
    reduction step.
  - `n_par::NumericalParameters`: numerical parameters (provides DCT sizes, etc.).
  - `m_par::ModelParameters`: model parameters (provides utility parameters, etc.).

# Returns

    - `ind::Union{Array{Array{Int}}, Array{Int}}`: selected DCT coefficient indices by asset.
    - `J::AbstractMatrix{Float64}`: Jacobian of marginal values with respect to selected inputs.
"""
function first_stage_reduction(
    vfSS::Union{ValueFunctionsOneAsset,ValueFunctionsTwoAssets},
    transition_matricesSS::TransitionMatrices,
    pfSS::PolicyFunctions,
    args_hh_prob,
    include_list_idx,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    ## ------------------------------------------------------------------------------------
    ## Step 1: Calculate the Jacobian of the marginal value functions
    ## ------------------------------------------------------------------------------------

    # Define the Jacobian of the marginal value function with respect to `inputsSS`
    Jaux = ForwardDiff.jacobian(p -> VFI(p, vfSS, n_par, m_par), log.(args_hh_prob))

    # Scale the Jacobian of σ (hard-coded)
    σ_idx = findfirst(x -> x == "σ", args_hh_prob_names)
    Jaux[:, σ_idx] .= Jaux[:, σ_idx] .* 1500

    # Remove the co-linear variables in arguments of household problem
    J = Jaux[:, include_list_idx]

    # Step 2 + 3:
    # - Calculate the derivatives of policy functions as central finite differences
    # - Construct the joint transition matrix GammaTilde
    GammaTilde = construct_transition_matrix(pfSS, transition_matricesSS, n_par, m_par)

    ## ------------------------------------------------------------------------------------
    ## Step 4: Solve for Jacobian using the recursive relation
    ## ------------------------------------------------------------------------------------

    # Assumption: approximate autocorrelation of prices
    phi = 0.999

    # Solve the system
    W = (I - phi .* GammaTilde) \ J
    sizeJ = size(J, 2)

    ## ------------------------------------------------------------------------------------
    ## Step 5: Transform the marginal values using the inverse marginal utility function
    ## ------------------------------------------------------------------------------------

    CBarHat = transform_marginal_values(W, vfSS, n_par, m_par)

    ## ------------------------------------------------------------------------------------
    ## Step 6: Perform the selection of DCT coefficients to fit the transformation well
    ## ------------------------------------------------------------------------------------

    ind = select_DCT_indices(CBarHat, vfSS, n_par, m_par, sizeJ)

    return ind, J
end

function first_stage_reduction(
    vfSS::ValueFunctionsCompleteMarkets,
    transition_matricesSS::TransitionMatrices,
    pfSS::PolicyFunctions,
    args_hh_prob,
    include_list_idx,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    indb = [1 2]
    indk = [1 2]
    return []
end

"""
    VFI(args_hh_prob, vfSS, n_par, m_par)

Internal helper used by the reduction step. Given aggregate inputs `args_hh_prob`, computes
the vectorized steady-state marginal values by:

  - reading/writing household problem arguments via parsing macros,
  - computing net incomes (`incomes`) and expected marginal values,
  - updating policies with EGM (`EGM_policyupdate`), and
  - constructing updated marginal values (`updateW`).

Returns a vector concatenating the marginal values needed for Jacobian evaluation.
"""
function VFI(
    args_hh_prob,
    vfSS::ValueFunctions,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    args_hh_prob = exp.(args_hh_prob)

    @read_args_hh_prob()

    # Additional definitions: Human capital transition
    Π = n_par.Π .+ zeros(eltype(args_hh_prob), 1)[1]
    PP = ExTransition(m_par.ρ_h, n_par.bounds_h, sqrt(σ))
    if n_par.entrepreneur
        Π[1:(end - 1), 1:(end - 1)] = PP .* (1.0 - m_par.ζ)
    end

    # Net incomes of households
    @write_args_hh_prob()
    net_income, _, eff_int = incomes(n_par, m_par, args_hh_prob)

    # Expected marginal values
    EvfPrime = expected_marginal_values(Π, vfSS, n_par)

    # Calculate optimal policies
    pf = EGM_policyupdate(
        EvfPrime,
        args_hh_prob,
        net_income,
        n_par,
        m_par,
        n_par.warn_egm,
        n_par.model,
    )

    # Update marginal values
    vf_up = updateW(EvfPrime, pf, args_hh_prob, m_par, n_par)

    vf_up.b .*= eff_int

    return reduce(vcat, vec.(struc_to_vec(vf_up)))
end

"""
    construct_transition_matrix(pfSS, transition_matricesSS, n_par, m_par)

Build the joint linear transition operator Γ̃ used in the recursive Jacobian relation by
combining central-difference derivatives of steady-state policy functions with exogenous
transition matrices. The two-asset method assembles a block operator over `(b,k)` policies
and both exogenous transitions (productivity and market participation). The one-asset method
reduces to the liquid-asset policy derivative times the exogenous transition Γ.
"""
function construct_transition_matrix(
    pfSS::PolicyFunctionsTwoAssets,
    transition_matricesSS::TransitionMatricesTwoAssets,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    ## ------------------------------------------------------------------------------------
    ## Step 2: Calculate the derivatives of policy functions as central finite differences
    ## ------------------------------------------------------------------------------------

    Dk_ak = spdiagm(
        m_par.β .* m_par.λ .* centralderiv(
            reshape(pfSS.k_a_star, (n_par.nb, n_par.nk, n_par.nh)),
            n_par.mesh_k,
            2,
        )[:],
    )
    Db_ak = spdiagm(
        m_par.β .* m_par.λ .* centralderiv(
            reshape(pfSS.b_a_star, (n_par.nb, n_par.nk, n_par.nh)),
            n_par.mesh_k,
            2,
        )[:],
    )
    Dk_ab = spdiagm(
        m_par.β .* m_par.λ .* centralderiv(
            reshape(pfSS.k_a_star, (n_par.nb, n_par.nk, n_par.nh)),
            n_par.mesh_b,
            1,
        )[:],
    )
    Db_ab = spdiagm(
        m_par.β .* m_par.λ .* centralderiv(
            reshape(pfSS.b_a_star, (n_par.nb, n_par.nk, n_par.nh)),
            n_par.mesh_b,
            1,
        )[:],
    )
    Dk_nk = m_par.β .* (1.0 .- m_par.λ)
    Db_nk = spdiagm(
        m_par.β .* (1.0 .- m_par.λ) .* centralderiv(
            reshape(pfSS.b_n_star, (n_par.nb, n_par.nk, n_par.nh)),
            n_par.mesh_k,
            2,
        )[:],
    )
    Dk_nb = 0.0
    Db_nb = spdiagm(
        m_par.β .* (1.0 .- m_par.λ) .* centralderiv(
            reshape(pfSS.b_n_star, (n_par.nb, n_par.nk, n_par.nh)),
            n_par.mesh_b,
            1,
        )[:],
    )

    ## ------------------------------------------------------------------------------------
    ## Step 3: Construct the joint transition matrix
    ## ------------------------------------------------------------------------------------
    GammaTilde =
        [
            Db_ab*transition_matricesSS.Γ_a Dk_ab*transition_matricesSS.Γ_a
            Db_ak*transition_matricesSS.Γ_a Dk_ak*transition_matricesSS.Γ_a
        ] + [
            Db_nb*transition_matricesSS.Γ_n Dk_nb*transition_matricesSS.Γ_n
            Db_nk*transition_matricesSS.Γ_n Dk_nk*transition_matricesSS.Γ_n
        ]
    return GammaTilde
end

function construct_transition_matrix(
    pfSS::PolicyFunctionsOneAsset,
    transition_matricesSS::TransitionMatricesOneAsset,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    ## ------------------------------------------------------------------------------------
    ## Step 2: Calculate the derivatives of policy functions as central finite differences
    ## ------------------------------------------------------------------------------------

    Db_nb = spdiagm(
        m_par.β .* (1.0 .- m_par.λ) .*
        centralderiv(reshape(pfSS.b_n_star, (n_par.nb, n_par.nh)), n_par.mesh_b, 1)[:],
    )

    ## ------------------------------------------------------------------------------------
    ## Step 3: Construct the joint transition matrix
    ## ------------------------------------------------------------------------------------

    return Db_nb * transition_matricesSS.Γ
end

"""
    transform_marginal_values(W, vfSS, n_par, m_par)

Apply the outer derivative of the log inverse marginal utility map to convert Jacobians of
marginal values into derivatives of transformed consumption objects. Returns a two-element
vector `[CBarHat_b, CBarHat_k]` for two-asset models and a single-element vector
`[CBarHat_b]` for one-asset models.
"""
function transform_marginal_values(
    W,
    vfSS::ValueFunctionsTwoAssets,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    # Total grid size
    Numel = n_par.nb * n_par.nk * n_par.nh

    # Select Jacobian of marginal value function with respect to k and b
    Wb = W[1:Numel, :]
    Wk = W[(Numel + 1):(2 * Numel), :]

    # Obtain outer derivative of transformation
    TransformV(V) = log.(invmutil(V, m_par))
    Outerderivative(x) = ForwardDiff.derivative(V -> TransformV(V), x)

    # Apply chain rule to compute transformed derivatives
    CBarHat_k = Outerderivative.(vfSS.k[:]) .* Wk
    CBarHat_m = Outerderivative.(vfSS.b[:]) .* Wb

    return [CBarHat_m, CBarHat_k]
end

function transform_marginal_values(
    W,
    vfSS::ValueFunctionsOneAsset,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    # Total grid size
    Numel = n_par.nb * n_par.nh

    # Select Jacobian of marginal value function with respect to k and b
    @assert size(W, 1) == Numel "Size mismatch in W for OneAsset model."

    # Obtain outer derivative of transformation
    TransformV(V) = log.(invmutil(V, m_par))
    Outerderivative(x) = ForwardDiff.derivative(V -> TransformV(V), x)

    # Apply chain rule to compute transformed derivatives
    CBarHat_m = Outerderivative.(vfSS.b[:]) .* W

    return [CBarHat_m]
end

"""
    select_DCT_indices(CBarHat, vfSS, n_par, m_par, sizeJ)

Select DCT coefficients that best represent the (average absolute) derivatives of
transformed marginal values. For two-asset models, returns `[indb, indk]` combining
salience-based selection and steady-state shape-matching. For one-asset models with
`CBarHat`, returns `[indb]`. A fallback one-asset method selects from the DCT of the
(untransformed) value function when derivative information is not supplied.
"""
function select_DCT_indices(
    CBarHat,
    vfSS::ValueFunctionsTwoAssets,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    sizeJ::Int,
)
    # This step selects the coefficients that best match the average absolute derivative of
    # the marginal values in terms of their DCT representation.
    CBarHat_m = CBarHat[1]
    CBarHat_k = CBarHat[2]

    # Initialize index arrays
    indk = Array{Array{Int}}(undef, 2)
    indb = Array{Array{Int}}(undef, 2)

    # Apply the DCT to compute the transformation for CBarHat
    Theta_m = similar(CBarHat_m)
    Theta_k = similar(CBarHat_k)
    for j = 1:sizeJ
        Theta_m[:, j] = dct(reshape(CBarHat_m[:, j], (n_par.nb, n_par.nk, n_par.nh)))[:]
        Theta_k[:, j] = dct(reshape(CBarHat_k[:, j], (n_par.nb, n_par.nk, n_par.nh)))[:]
    end
    theta_m = sum(abs.(Theta_m); dims = 2)
    theta_k = sum(abs.(Theta_k); dims = 2)

    # Select DCT coefficients that explain the average derivative well
    indb[1] = select_ind(
        reshape(theta_m, (n_par.nb, n_par.nk, n_par.nh)),
        n_par.reduc_marginal_value,
    )
    indk[1] = select_ind(
        reshape(theta_k, (n_par.nb, n_par.nk, n_par.nh)),
        n_par.reduc_marginal_value,
    )

    # Add the DCT indices that match the shape of the marginal value functions well
    indk[end] = select_ind(
        dct(reshape(log.(invmutil(vfSS.k, m_par)), (n_par.nb, n_par.nk, n_par.nh))),
        n_par.reduc_value,
    )
    indb[end] = select_ind(
        dct(reshape(log.(invmutil(vfSS.b, m_par)), (n_par.nb, n_par.nk, n_par.nh))),
        n_par.reduc_value,
    )

    indb = sort(unique(vcat(indb...)))
    indk = sort(unique(vcat(indk...)))

    return [indb, indk]
end

function select_DCT_indices(
    CBarHat,
    vfSS::ValueFunctionsOneAsset,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    sizeJ::Int,
)
    # This step selects the coefficients that best match the average absolute derivative of
    # the marginal values in terms of their DCT representation.
    CBarHat_m = CBarHat[1]

    # Initialize index arrays
    indb = Array{Array{Int}}(undef, 2)

    # Apply the DCT to compute the transformation for CBarHat
    Theta_m = similar(CBarHat_m)
    for j = 1:sizeJ
        Theta_m[:, j] = dct(reshape(CBarHat_m[:, j], (n_par.nb, n_par.nk, n_par.nh)))[:]
    end
    theta_m = sum(abs.(Theta_m); dims = 2)

    # Select DCT coefficients that explain the average derivative well
    indb[1] = select_ind(
        reshape(theta_m, (n_par.nb, n_par.nk, n_par.nh)),
        n_par.reduc_marginal_value,
    )

    # Add the DCT indices that match the shape of the marginal value functions well
    indb[end] = select_ind(
        dct(reshape(log.(invmutil(vfSS.b, m_par)), (n_par.nb, n_par.nk, n_par.nh))),
        n_par.reduc_value,
    )

    indb = sort(unique(vcat(indb...)))
    return [indb]
end

function select_DCT_indices(vfSS::ValueFunctionsOneAsset, n_par::NumericalParameters)
    # Apply the DCT to compute the transformation for value function
    theta_b = dct(vfSS.b)

    # Select DCT coefficients that explain the marginal value function well
    indb = select_ind(reshape(theta_b, (n_par.nb, n_par.nh)), n_par.reduc_marginal_value)
    # return [sort(indb)]
    return [indb]
end
