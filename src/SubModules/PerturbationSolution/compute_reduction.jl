"""
    compute_reduction(sr, lr, m_par, shock_names)

Compute the second-stage model reduction using Principal Component Analysis (PCA) on the
covariance matrices of the linearized model's states and controls.

This function identifies the linear combinations of the first-stage variables (DCT
coefficients of marginal value functions and distributions) that contribute most to the
model's volatility over the business cycle. It constructs projection matrices (`PRightAll`,
`PRightStates`) to project the full system onto this reduced subspace, significantly
reducing the number of state and control variables for estimation while maintaining
approximation quality.

# Arguments

  - `sr`: Steady state results containing indices and parameters.
  - `lr`: Initial linear solution (first-stage reduction) used to compute long-run
    covariances.
  - `m_par`: Model parameters (used for shock variances).
  - `shock_names`: Vector of symbols denoting the shocks to include in the covariance
    calculation.

# Returns

  - `indexes_r`: Structure containing indices for the reduced model variables.

  - `n_par`: Updated numerical parameters object containing:

      + Reduced dimensions (`nstates_r`, `ncontrols_r`, `ntotal_r`).
      + Projection matrices (`PRightAll`, `PRightStates`) mapping full variables to reduced
        factors.
"""
function compute_reduction(sr, lr, m_par, shock_names)

    #---------------------------------------------------------------
    ## STEP 1: PRODUCE Long Run Covariance
    #---------------------------------------------------------------
    n_par = sr.n_par
    SCov = zeros(n_par.nstates, n_par.nstates)
    for i in shock_names
        SCov[getfield(sr.indexes, i), getfield(sr.indexes, i)] =
            (getfield(m_par, Symbol("σ_", i))) .^ 2
    end

    StateCOVAR = lyapd(lr.LOMstate, SCov)

    ControlCOVAR = lr.State2Control * StateCOVAR * lr.State2Control'
    ControlCOVAR = (ControlCOVAR + ControlCOVAR') ./ 2
    #---------------------------------------------------------------
    ## STEP 1: Produce eigenvalue decomposition of relevant COVARs
    #          and select eigenvectors of large eigenvalues to obtain
    #          low-dim to high-dim projections of copula and value-functions
    #---------------------------------------------------------------
    # Dindex       = sr.indexes.distr_b
    # evalS, evecS = eigen(StateCOVAR[Dindex,Dindex])
    # println(sum(abs.(evalS).> maximum(evalS)*n_par.further_compress_critS))
    # Dindex       = sr.indexes.distr_k
    # evalS, evecS = eigen(StateCOVAR[Dindex,Dindex])
    # println(sum(abs.(evalS).> maximum(evalS)*n_par.further_compress_critS))

    Dindex = get_Dindex(sr.indexes.distr)
    evalS, evecS = eigen(StateCOVAR[Dindex, Dindex])
    keepD = abs.(evalS) .> maximum(evalS) * n_par.further_compress_critS
    indKeepD = Dindex[keepD]
    @set! n_par.nstates_r = n_par.nstates - length(Dindex) + length(indKeepD)

    Vindex = [sr.indexes.valueFunction.b; sr.indexes.valueFunction.k]
    evalC, evecC = eigen(ControlCOVAR[Vindex .- n_par.nstates, Vindex .- n_par.nstates])
    keepV = abs.(evalC) .> maximum(evalC) * n_par.further_compress_critC
    indKeepV = Vindex[keepV]
    @set! n_par.ncontrols_r = n_par.ncontrols - length(Vindex) + length(indKeepV)

    #-------------------------------------------------------------
    ## Step 3: Put together projection matrices and update indexes
    #-------------------------------------------------------------

    PRightStates_aux = float(I[1:(n_par.nstates), 1:(n_par.nstates)])
    PRightStates_aux[Dindex, Dindex] = evecS
    keep = ones(Bool, n_par.nstates)
    keep[Dindex[.!keepD]] .= false
    @set! n_par.PRightStates = PRightStates_aux[:, keep]

    PRightAll_aux = float(I[1:(n_par.ntotal), 1:(n_par.ntotal)])
    PRightAll_aux[Dindex, Dindex] = evecS
    PRightAll_aux[Vindex, Vindex] = evecC
    keep = ones(Bool, n_par.ntotal)
    keep[Dindex[.!keepD]] .= false
    keep[Vindex[.!keepV]] .= false
    # @set! n_par.PRightAll        = PRightAll_aux[:,keep]
    Aux = PRightAll_aux[:, keep]
    # TODO: Here, we have to adjust the indexes to the new version with both value function indices in one vector – similar to prepare_linearization! DONE, but move to a function?
    indexes_r =
        produce_indexes(n_par, [keepV[keepV][1:2], keepV[keepV][3:end]], keepD[keepD]) # arbitrary splitup of underlying factors from value functions to indexes in reduced model
    @set! n_par.ntotal_r = n_par.nstates_r + n_par.ncontrols_r

    # TODO: This is highly order sensitive, I have adjusted it but we need to make this work for all solution methods – move to a function?
    block = Array{Vector}(undef, 5)
    block_r = Array{Vector}(undef, 5)
    indexes = sr.indexes
    block[1] = indexes.distr.COP
    block[2] = [indexes.distr.b; indexes.distr.k; indexes.distr.h]
    block[3] = (indexes.distr.h[end] + 1):(indexes.valueFunction.b[1] - 1)
    block[4] = [indexes.valueFunction.b; indexes.valueFunction.k]
    block[5] = (indexes.valueFunction.k[end] + 1):(n_par.ntotal)
    block_r[1] = indexes_r.distr.COP
    block_r[2] = [indexes_r.distr.b; indexes_r.distr.k; indexes_r.distr.h]
    block_r[3] = (indexes_r.distr.h[end] + 1):(indexes_r.valueFunction.b[1] - 1)
    block_r[4] = [indexes_r.valueFunction.b; indexes_r.valueFunction.k]
    block_r[5] = (indexes_r.valueFunction.k[end] + 1):(n_par.ntotal_r)
    RedCOP = Aux[block[1], block_r[1]]
    RedVs = Aux[block[4], block_r[4]]
    @set! n_par.PRightAll = BlockDiagonal([
        RedCOP,                         # compression for copula
        diagm(ones(length(block[2]))),    # mapping from marginal dist to marginal dist
        diagm(ones(length(block[3]))),  # mapping aggr states to aggr states
        RedVs,                          # compression for value functions
        diagm(ones(length(block[5]))),
    ]) # mapping aggr controls to aggr controls
    #println(Matrix(n_par.PRightAll)==Aux)

    return indexes_r, n_par
end
