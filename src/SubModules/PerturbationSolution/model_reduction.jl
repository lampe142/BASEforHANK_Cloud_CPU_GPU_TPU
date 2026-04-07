"""
    model_reduction(sr, lr, m_par)

Produce Model Reduction based on Variance Covariance Matrix of States and Controls.

This function performs the second stage of model reduction. It computes the covariance of
the states and controls using the first-stage solution (linearized model `lr`) and then uses
Principal Component Analysis (PCA) via `compute_reduction` to find a lower-dimensional
subspace that captures the most significant dynamics.

The reduction is based on the method described in Bayer, Born, and Luetticke (2024, AER). If
`n_par.further_compress` is `true`, it updates the projection matrices in `n_par` to map the
full state space to this reduced factor space.

# Arguments

  - `sr::SteadyResults`: Steady state results containing initial indices and parameters.
  - `lr::LinearResults`: Linear solution from the first stage (DCT reduction only).
  - `m_par::ModelParameters`: Model parameters.

# Returns

  - `SteadyResults`: A new `SteadyResults` struct containing:

      + Updated `indexes_r` (indices for the reduced model).
      + Updated `n_par` (numerical parameters with reduced dimensions and projection
        matrices).
      + Other fields copied from the input `sr`.
"""
function model_reduction(sr, lr, m_par)
    @printf "\n"
    @printf "Model reduction (state-space representation)...\n"

    n_par = sr.n_par

    if n_par.further_compress
        @printf "Reduction Step\n"
        indexes_r, n_par = compute_reduction(sr, lr, m_par, e_set.shock_names)

        @printf "Number of reduced model factors for DCTs for Wb & Wk: %d\n" (
            length(indexes_r.valueFunction.b) + length(indexes_r.valueFunction.k)
        )

        @printf "Number of reduced model factors for copula DCTs: %d\n" length(
            indexes_r.distr.COP,
        )
    else
        @printf "Further model reduction switched off --> reverting to full model\n"
        @set! n_par.PRightAll = Diagonal(ones(n_par.ntotal))
        @set! n_par.PRightStates = Diagonal(ones(n_par.nstates))
        indexes_r = sr.indexes
        @set! n_par.nstates_r = n_par.nstates
        @set! n_par.ncontrols_r = n_par.ncontrols
        @set! n_par.ntotal_r = n_par.ntotal
    end

    @printf "Model reduction (state-space representation)... Done.\n"

    return SteadyResults(
        sr.XSS,
        sr.XSSaggr,
        sr.indexes,
        indexes_r,
        sr.indexes_aggr,
        sr.compressionIndexes,
        n_par,
        m_par,
        sr.distrSS,
        state_names,
        control_names,
    )
end
