"""
    Ksupply(
        args_hh_prob,
        n_par,
        m_par,
        vf,
        distr_guess,
        net_income,
        eff_int,
    )

Calculate the aggregate savings when households face idiosyncratic income risk by first
solving the household problem using the endogenous grid method (EGM) and then finding the
stationary distribution of idiosyncratic states.

Idiosyncratic state is tuple `(b,k,y)`, where:

 1. `b`: liquid assets,
 2. `k`: illiquid assets,
 3. `y`: labor income.

This function is used in [`find_steadystate()`](@ref) to find the stationary equilibrium, as
an input to [`Kdiff()`](@ref), and in
[`BASEforHANK.PerturbationSolution.prepare_linearization()`](@ref) to prepare the
linearization of the model.

# Arguments

  - `args_hh_prob`: Vector of arguments to the household problem
  - `n_par`: Numerical parameters
  - `m_par`: Model parameters
  - `vf`: Initial guess for marginal value functions (ValueFunctions struct)
  - `distr_guess`: Initial guess for stationary distribution
  - `net_income`: Net incomes, output of functions from the IncomesETC module
  - `eff_int`: Effective interest rates, output of functions from the IncomesETC module

# Returns

  - `K`: Aggregate saving in illiquid assets
  - `B`: Aggregate saving in liquid assets
  - `transition_matrices`: Transition matrices struct containing sparse transition matrices
  - `pf`: Optimal policy functions (PolicyFunctions struct)
  - `vf`: Updated marginal value functions (ValueFunctions struct)
  - `distr`: Ergodic stationary distribution
"""
function Ksupply(
    args_hh_prob::Vector,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    vf::ValueFunctions,
    distr_guess::AbstractArray,
    net_income::AbstractArray,
    eff_int::AbstractArray,
)

    ## ------------------------------------------------------------------------------------
    ## Step 1: Find optimal policies
    ## ------------------------------------------------------------------------------------

    pf, vf = find_ss_policies(args_hh_prob, n_par, m_par, vf, net_income, eff_int)

    ## ------------------------------------------------------------------------------------
    ## Step 2: Find stationary distribution
    ## ------------------------------------------------------------------------------------

    distr, transition_matrices =
        find_ss_distribution(pf, distr_guess, m_par, n_par, n_par.transition_type)

    ## ------------------------------------------------------------------------------------
    ## Step 3: Calculate aggregate savings
    ## ------------------------------------------------------------------------------------
    B, K = aggregate_B_K(distr, m_par, n_par, n_par.model)

    ## ------------------------------------------------------------------------------------
    ## Return results
    ## ------------------------------------------------------------------------------------

    return K, B, transition_matrices, pf, vf, distr
end

"""
    find_ss_policies(args_hh_prob, n_par, m_par, vf::ValueFunctionsCompleteMarkets, net_income, eff_int)

Find steady-state policy functions for the complete markets model.

For complete markets, no policy iteration is needed.

# Arguments

  - `args_hh_prob`: Vector of arguments to the household problem (unused for complete
    markets)
  - `n_par`: Numerical parameters
  - `m_par`: Model parameters
  - `vf`: Initial value functions (unused for complete markets)
  - `net_income`: Net incomes (unused for complete markets)
  - `eff_int`: Effective interest rates

# Returns

  - `pf`: Empty policy functions struct for complete markets
  - `vf`: Value functions struct with effective interest rates
"""
function find_ss_policies(
    args_hh_prob::Vector,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    vf::ValueFunctionsCompleteMarkets,
    net_income::AbstractArray,
    eff_int::AbstractArray,
)
    return PolicyFunctionsCompleteMarkets{typeof(vf.b)}(),
    ValueFunctionsCompleteMarkets(ones(eltype(vf.b), size(vf.b)) .* eff_int)
end

"""
    find_ss_policies(args_hh_prob, n_par, m_par, vf::ValueFunctionsOneAsset, net_income, eff_int)

Find steady-state policy functions for the one-asset model using value function iteration.

Solves the household problem for agents with only liquid assets using the endogenous grid
method (EGM) until the marginal value functions converge.

# Arguments

  - `args_hh_prob`: Vector of arguments to the household problem
  - `n_par`: Numerical parameters
  - `m_par`: Model parameters
  - `vf`: Initial guess for marginal value functions
  - `net_income`: Net incomes from the IncomesETC module
  - `eff_int`: Effective interest rates

# Returns

  - `pf`: Optimal policy functions for the one-asset model
  - `vf`: Converged marginal value functions
"""
function find_ss_policies(
    args_hh_prob::Vector,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    vf::ValueFunctionsOneAsset,
    net_income::AbstractArray,
    eff_int::AbstractArray,
)

    ## ------------------------------------------------------------------------------------
    ## Step 1: Preallocate variables
    ## ------------------------------------------------------------------------------------

    pf = PolicyFunctionsOneAsset{typeof(vf.b)}(vf.b)
    Wb = vf.b

    dist = 9999.0
    count = 0

    nb, nh = size(Wb)

    # Expected marginal value functions
    EWb = similar(Wb)

    # New marginal value functions
    Wb_new = similar(Wb)

    # Inverse marginal value functions
    iWb = invmutil(Wb, m_par)
    iWb_new = similar(iWb)

    # Distance between old and new inverse policy functions
    D = similar(Wb)

    # Difference between expected marginal value functions of assets
    E_return_diff = similar(Wb)

    # Marginal utility of consumption
    EMU = similar(Wb)

    # Marginal utility of consumption, adjustment case
    mutil_x_a = similar(Wb)

    ## Resouce grid -----------------------------------------------------------------------

    # Asset income plus liquidation value (adjustment case)
    n_rental_inc = net_income[2]
    liquid_asset_inc = net_income[3]
    capital_liquidation_inc = net_income[4]

    # Exogenous resource grid for the adjustment case in EGM calculated according to eq.
    # (resources adjustment)
    R_exo_a = n_rental_inc .+ liquid_asset_inc .+ capital_liquidation_inc

    ## ------------------------------------------------------------------------------------
    ## Step 2: Loop
    ## ------------------------------------------------------------------------------------

    # Iterate over marginal values until convergence
    loop_time = time()
    while dist > n_par.ϵ && count < 10000
        count += 1

        ## Take expectations given exogenous state transition -----------------------------

        # Repeating the same process for the expected marginal value of liquid assets
        # following eq. (ECV2)
        BLAS.gemm!('N', 'T', 1.0, Wb, n_par.Π, 0.0, EWb)

        # Applying the effective interest rate to the expected marginal value of liquid
        # assets to match the expression shown in the Envelope condition in the
        # documentation (CV2)
        EWb .*= eff_int

        vf.b .= EWb

        ## Policy update step -------------------------------------------------------------
        # Given expected marginal values, update the policy functions
        EGM_policyupdate!(
            pf,
            E_return_diff,
            EMU,
            R_exo_a,
            vf,
            args_hh_prob,
            net_income,
            n_par,
            m_par,
            n_par.warn_egm,
        )

        ## Marginal value update step -------------------------------------------------------

        # Compute marginal utility of the composite, non-adjustment case
        mutil!(Wb_new, pf.x_n_star, m_par)

        ## Calculate distance in updates and update ----------------------------------------

        # Calculate distance of inverse marginal value functions (more conservative), and
        # update marginal value functions

        invmutil!(iWb_new, Wb_new, m_par)

        D .= iWb_new .- iWb
        dist = maximum(abs, D)
        Wb .= Wb_new
        iWb .= iWb_new
    end
    loop_time = time() - loop_time

    if n_par.verbose
        @printf "EGM: Iterations %d, Distance %.2e, Time %.2f\n" count dist loop_time
    end

    vf = ValueFunctionsOneAsset(Wb)
    return pf, vf
end

"""
    find_ss_policies(args_hh_prob, n_par, m_par, vf::ValueFunctionsTwoAssets, net_income, eff_int)

Find steady-state policy functions for the two-asset model using value function iteration.

Solves the household problem for agents with both liquid and illiquid assets using the
endogenous grid method (EGM) until the marginal value functions converge. Handles both
adjustment and non-adjustment cases for illiquid assets.

# Arguments

  - `args_hh_prob`: Vector of arguments to the household problem
  - `n_par`: Numerical parameters
  - `m_par`: Model parameters
  - `vf`: Initial guess for marginal value functions (liquid and illiquid assets)
  - `net_income`: Net incomes from the IncomesETC module
  - `eff_int`: Effective interest rates

# Returns

  - `pf`: Optimal policy functions for the two-asset model
  - `vf`: Converged marginal value functions for both assets
"""
function find_ss_policies(
    args_hh_prob::Vector,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    vf::ValueFunctionsTwoAssets,
    net_income::AbstractArray,
    eff_int::AbstractArray,
)
    Wb = vf.b
    Wk = vf.k

    ## ------------------------------------------------------------------------------------
    ## Step 1: Preallocate variables
    ## ------------------------------------------------------------------------------------

    ## Loop variables ---------------------------------------------------------------------

    dist = 9999.0
    dist1 = dist
    dist2 = dist
    count = 0

    ## Policy and value functions ---------------------------------------------------------

    nb, nk, nh = size(Wb)

    # Policy functions
    pf = PolicyFunctionsTwoAssets{typeof(Wb)}(Wb)

    # Expected marginal value functions
    Evf = similar(vf)
    EWb = similar(Wb) # required additionally due to reshaping
    EWk = similar(Wk)

    # New marginal value functions
    vf_new = similar(vf)

    # Inverse marginal value functions
    ivf = similar(vf)
    ivf.b = invmutil(vf.b, m_par)
    ivf.k = invmutil(vf.k, m_par)
    ivf_new = similar(ivf)

    # Distance between old and new inverse policy functions
    D1 = similar(Wb)
    D2 = similar(Wb)

    # Difference between expected marginal value functions of assets
    E_return_diff = similar(Wb)

    # Marginal utility of consumption
    EMU = similar(Wb)

    # Marginal utility of consumption, adjustment case
    mutil_x_a = similar(Wb)

    ## Resouce grid -----------------------------------------------------------------------

    # Asset income plus liquidation value (adjustment case)
    n_rental_inc = net_income[2]
    liquid_asset_inc = net_income[3]
    capital_liquidation_inc = net_income[4]

    # Exogenous resource grid for the adjustment case in EGM calculated according to eq.
    # (resources adjustment)
    R_exo_a =
        reshape(n_rental_inc .+ liquid_asset_inc .+ capital_liquidation_inc, (nb .* nk, nh))

    ##  Precomputed reshaped views --------------------------------------
    # Create persistent reshaped views outside the loop to eliminate repeated reshape
    # operations in BLAS calls
    Wk_flat = reshape(Wk, (nb * nk, nh))
    EWk_flat = reshape(EWk, (nb * nk, nh))
    Wb_flat = reshape(Wb, (nb * nk, nh))
    EWb_flat = reshape(EWb, (nb * nk, nh))

    # Pre-allocate 3D views
    EWk_3d = reshape(EWk_flat, (nb, nk, nh))
    EWb_3d = reshape(EWb_flat, (nb, nk, nh))

    ## ------------------------------------------------------------------------------------
    ## Step 2: Loop
    ## ------------------------------------------------------------------------------------

    # Iterate over marginal values until convergence
    loop_time = time()
    while dist > n_par.ϵ && count < 10000
        count += 1

        ## Take expectations given exogenous state transition -----------------------------

        #=
        Perform matrix multiplication using BLAS and reshape the result.

        Essentially this function calculates the expected marginal value of illiquid assets
        as would EWk .= reshape(reshape(Wk, (nb .* nk, nh)) * n_par.Π', (nb, nk, nh)),
        however, the BLAS `gemm!` function allows faster computation.

        The following lines use the BLAS `gemm!` function to perform fast matrix
        multiplication on the input matrix `Wk` and the parameter matrix `n_par.Π`. The
        result is stored in `EWk`. The matrices are reshaped before and after the
        multiplication to match the required dimensions.

        Arguments:
        - `Wk`: Input matrix to be multiplied, reshaped to dimensions `(nb * nk, nh)`.
        - `n_par.Π`: Parameter matrix used for multiplication.
        - `EWk`: Output matrix to store the result, reshaped to dimensions `(nb * nk, nh)`.
        - `n`: Tuple containing the dimensions for reshaping the matrices.

        Additionally, BLAS.gemm! has inputs 1.0 and 0.0 to specify potential scaling factors
        for the input matrices (not used). The arguments 'N' and 'T' specify that the
        matrices are not transposed before multiplication.

        The final result in `EWk` is reshaped back to dimensions `(nb, nk, nh)`.
        =#

        # Calculate expected marginal value of illiquid assets using precomputed views
        # (ECV1)
        BLAS.gemm!('N', 'T', 1.0, Wk_flat, n_par.Π, 0.0, EWk_flat)
        # EWk_3d is automatically updated since it's a view of EWk_flat
        Evf.k .= EWk_3d  # Use .= instead of copy() to avoid allocation

        # Calculate expected marginal value of liquid assets using precomputed views (ECV2)
        BLAS.gemm!('N', 'T', 1.0, Wb_flat, n_par.Π, 0.0, EWb_flat)
        # EWb_3d is automatically updated since it's a view of EWb_flat

        # Applying the effective interest rate to the expected marginal value of liquid
        # assets to match the expression shown in the Envelope condition in the
        # documentation (CV2)
        EWb_3d .*= eff_int
        Evf.b .= EWb_3d

        ## Policy update step -------------------------------------------------------------

        # Given expected marginal values, update the policy functions
        EGM_policyupdate!(
            pf,
            E_return_diff,
            EMU,
            R_exo_a,
            Evf,
            args_hh_prob,
            net_income,
            n_par,
            m_par,
            n_par.warn_egm,
        )

        ## Marginal value update step -------------------------------------------------------

        # Given the policy functions, update the marginal value functions
        updateW!(vf_new, mutil_x_a, Evf, pf, args_hh_prob, m_par, n_par)

        ## Calculate distance in updates and update ----------------------------------------

        # Calculate distance of inverse marginal value functions (more conservative), and
        # update marginal value functions

        invmutil!(ivf_new.k, vf_new.k, m_par)
        invmutil!(ivf_new.b, vf_new.b, m_par)

        D1 .= ivf_new.k .- ivf.k
        D2 .= ivf_new.b .- ivf.b

        dist1 = maximum(abs, D1)
        dist2 = maximum(abs, D2)

        dist = max(dist1, dist2)

        Wk .= vf_new.k
        Wb .= vf_new.b
        ivf.k .= ivf_new.k
        ivf.b .= ivf_new.b
    end
    loop_time = time() - loop_time

    if n_par.verbose
        @printf "EGM: Iterations %d, Distance %.2e, Time %.2f\n" count dist loop_time
    end

    vf = ValueFunctionsTwoAssets(Wb, Wk)

    return pf, vf
end

"""
    find_transitions(pf::PolicyFunctionsTwoAssets, n_par, m_par)

Construct transition matrices for the two-asset model based on optimal policy functions.

Creates sparse transition matrices for both adjustment and non-adjustment cases of illiquid
assets, then combines them using the adjustment probability λ.

# Arguments

  - `pf`: Policy functions for the two-asset model
  - `n_par`: Numerical parameters
  - `m_par`: Model parameters (including adjustment probability λ)

# Returns

  - `TransitionMatricesTwoAssets`: Struct containing joint transition matrix Γ, adjustment
    transition matrix Γ_a, and non-adjustment transition matrix Γ_n
"""
function find_transitions(
    pf::PolicyFunctionsTwoAssets,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    ## Define transition matrix -----------------------------------------------------------

    # Calculate inputs for sparse transition matrix
    #=

    MakeTransition from fcn_maketransition.jl returns the following variables:
    - S_a: start index for adjustment case
    - T_a: target index for adjustment case
    - W_a: weight for adjustment case
    - S_n: start index for non-adjustment case
    - T_n: target index for non-adjustment case
    - W_n: weight for non-adjustment case

    =#
    S_a, T_a, W_a, S_n, T_n, W_n = MakeTransition(pf, n_par.Π, n_par)

    # Create sparse transition matrix for adjustment case of dimensions nb * nk * nh times
    # nb * nk * nh such that Γ_a[S_a[k], T_a[k]] = W_a[k].
    Γ_a = sparse(
        S_a,
        T_a,
        W_a,
        n_par.nb * n_par.nk * n_par.nh,
        n_par.nb * n_par.nk * n_par.nh,
    )

    # Create sparse transition matrix for non-adjustment case of dimensions nb * nk * nh
    # times nb * nk * nh such that Γ_n[S_n[k], T_n[k]] = W_n[k].
    Γ_n = sparse(
        S_n,
        T_n,
        W_n,
        n_par.nb * n_par.nk * n_par.nh,
        n_par.nb * n_par.nk * n_par.nh,
    )

    # Joint, probability-weighted transition matrix
    Γ = m_par.λ .* Γ_a .+ (1.0 .- m_par.λ) .* Γ_n
    return TransitionMatricesTwoAssets(Γ, Γ_a, Γ_n)
end

"""
    find_transitions(pf::PolicyFunctionsOneAsset, n_par, m_par)

Construct transition matrix for the one-asset model based on optimal policy functions.

Creates a sparse transition matrix based on the liquid asset policy functions.

# Arguments

  - `pf`: Policy functions for the one-asset model
  - `n_par`: Numerical parameters
  - `m_par`: Model parameters

# Returns

  - `TransitionMatricesOneAsset`: Struct containing the transition matrix Γ
"""
function find_transitions(
    pf::PolicyFunctionsOneAsset,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    ## Define transition matrix -----------------------------------------------------------

    # Calculate inputs for sparse transition matrix
    #=

    MakeTransition from fcn_maketransition.jl returns the following variables:
    - S_n: start index
    - T_n: target index
    - W_n: weight

    =#
    S_n, T_n, W_n = MakeTransition(pf, n_par.Π, n_par)

    # Create sparse transition matrix
    Γ = sparse(
        S_n,
        T_n,
        W_n,
        n_par.nb * n_par.nk * n_par.nh,
        n_par.nb * n_par.nk * n_par.nh,
    )

    return TransitionMatricesOneAsset(Γ)
end

"""
    find_ss_distribution(pf, distr_guess, m_par, n_par)

Find the stationary distribution for one-asset or two-asset models.

Computes the steady-state distribution by finding the left-hand unit eigenvector of the
transition matrix.

# Arguments

  - `pf`: Policy functions (OneAsset or TwoAssets)
  - `distr_guess`: Initial guess for the distribution
  - `m_par`: Model parameters
  - `n_par`: Numerical parameters

# Returns

  - `distr`: Stationary distribution normalized to sum to 1
  - `transition_matrices`: Transition matrices struct
"""
function find_ss_distribution(
    pf::Union{PolicyFunctionsOneAsset,PolicyFunctionsTwoAssets},
    distr_guess::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::LinearTransition,
)
    size_dims = size(distr_guess)

    # Find transition matrices
    transition_matrices = find_transitions(pf, n_par, m_par)

    ## Stationary distribution ------------------------------------------------------------

    # Calculate left-hand unit eigenvector of Γ' using eigsolve from KrylovKit
    aux = real.(eigsolve(transition_matrices.Γ', distr_guess[:], 1)[2][1])

    # Normalize and reshape to stationary distribution (nb x nk x nh)
    distr = (reshape((aux[:]) ./ sum((aux[:])), size_dims))

    return distr, transition_matrices
end

"""
    find_ss_distribution_splines( m_n_star::AbstractArray, cdf_guess_intial::AbstractArray,
        n_par::NumericalParameters, mbar::AbstractArray )

    Find steady state disribution given steady state policies and guess for the interest
        rate through iteration. Iteration of distribution is done using monotonic spline
        interpolation to bring the next periods cdf's back to the reference grid.

    # Arguments
    - `m_n_star::AbstractArray`: optimal savings choice given fixed grid of assets and
      income shocks
    - `cdf_guess_intial::AbstractArray`: initial guess for stationary cumulative joint
      distribution (in m) `cdf_guess[m,y]
    - `n_par::NumericalParameters`


    # Returns
    - `pdf_ss`: stationary distribution over m and y
"""
function find_ss_distribution(
    pf::PolicyFunctionsOneAsset,
    distr_guess::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::NonLinearTransition,
)

    # Tolerance for change in cdf from period to period
    tol = n_par.ϵ
    # Maximum iterations to find steady state distribution
    max_iter = 50000

    # Calculate cdf over individual income states
    cdf_guess = cumsum(copy(distr_guess); dims = 1)

    # Iterate on distribution until convergence
    distance = 9999.0
    iter = 0
    while distance > tol && iter < max_iter
        if iter % 10000 == 0 && iter > 0
            println(" -- Distribution Iterations: ", iter)
            println(" -- Distribution Dist: ", distance)
        end
        iter = iter + 1
        cdf_guess_old = copy(cdf_guess)
        DirectTransition!(
            cdf_guess,
            pf,
            cdf_guess_old,
            m_par.λ,
            n_par.Π,
            n_par,
            transition_type,
        )
        difference = cdf_guess_old .- cdf_guess
        distance = maximum(abs, difference)
    end

    pdf_ss = copy(cdf_guess)
    pdf_ss[2:end, :] .= diff(pdf_ss; dims = 1)

    if n_par.verbose
        println("Distribution Iterations: ", iter)
        println("Distribution Dist: ", distance)
    end

    return pdf_ss, TransitionMatricesOneAsset(sparse(I(2)))
end

function find_ss_distribution(
    pf::PolicyFunctionsTwoAssets,
    distr_guess::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::NonLinearTransition,
)
    error("NonLinearTransition for TwoAsset model currently in development.")
end

"""
    find_ss_distribution(pf::PolicyFunctionsCompleteMarkets, distr_guess, m_par, n_par)

Find the stationary distribution for the complete markets model. Set to income state PDF.

# Arguments

  - `pf`: Policy functions for complete markets (unused)
  - `distr_guess`: Initial guess for distribution (unused)
  - `m_par`: Model parameters (unused)
  - `n_par`: Numerical parameters (contains income transition matrix Π)

# Returns

  - `distr`: Stationary distribution of the income process
  - `transition_matrices`: Empty transition matrices struct for complete markets
"""
function find_ss_distribution(
    pf::PolicyFunctionsCompleteMarkets,
    distr_guess::AbstractArray,
    m_par::ModelParameters,
    n_par::NumericalParameters,
    transition_type::TransitionType,
)
    return (n_par.Π^1000)[1, :], TransitionMatricesCompleteMarkets(sparse(I(2)))
end
