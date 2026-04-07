"""
    update_model(sr, lr, m_par)

Efficiently update the linearized solution of the model when only aggregate parameters have
changed.

This function re-calculates the Jacobian of the aggregate equilibrium conditions
(`Fsys_agg`) and combines it with the previously computed Jacobian of the heterogeneous
agent block (which remains unchanged). It then solves the updated linear system using
`LinearSolution_reduced_system`. This provides a significant speedup during estimation loops
where only aggregate parameters are varied.

# Arguments

  - `sr::SteadyResults`: Steady state results containing the model structure and parameters.
  - `lr::LinearResults`: The previous linear solution results, containing the Jacobians `A`
    and `B` from the full model linearization.
  - `m_par::ModelParameters`: The new set of model parameters (with updated aggregate
    parameters).

# Returns

  - `LinearResults`: A new `LinearResults` struct containing:

      + `State2Control`: Updated observation matrix.
      + `LOMstate`: Updated state transition matrix.
      + `A`, `B`: Updated Jacobian matrices of the system.
      + `SolutionError`: Boolean flag indicating if the solution failed.
      + `nk`: Number of stable eigenvalues.
"""
function update_model(sr::SteadyResults, lr::LinearResults, m_par::ModelParameters)
    if sr.n_par.verbose
        @printf "Updating linearization\n"
    end
    State2Control, LOMstate, SolutionError, nk, A, B =
        LinearSolution_reduced_system(sr, m_par, lr.A, lr.B; allow_approx_sol = false)

    return LinearResults(State2Control, LOMstate, A, B, SolutionError, nk)
end
