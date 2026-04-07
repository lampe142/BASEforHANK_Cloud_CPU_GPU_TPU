"""
    compute_args_hh_prob_ss(K, m_par, n_par)

This function calculates the arguments (parameters, variables) that are needed for the
household problem, stored in `args_hh_prob` based on the list `args_hh_prob_names` from
`input_aggregate_names.jl`. These arguments will be packed and unpacked using the macros
[`@write_args_hh_prob`](@ref) (or [`@write_args_hh_prob_ss`](@ref)) and
[`@read_args_hh_prob`](@ref) (or [`@read_args_hh_prob_ss`](@ref)).

This function takes in the current capital stock `K` and the model and numerical parameters
as well as (user-specified) functions and then computes the relevant steady state values.
That means, all the necessary arguments must follow from `K` or steady state assumptions in
the parameter structs.

The function is used in q [`BASEforHANK.SteadyState.Kdiff()`](@ref) to find the stationary
equilibrium of the household block of the model and in
[`BASEforHANK.PerturbationSolution.prepare_linearization()`](@ref) to prepare the
linearization of the model.

# Arguments

  - `K::Float64`: Capital stock
  - `m_par::ModelParameters`, `n_par::NumericalParameters`

# Returns

  - `args_hh_prob::Vector`: Vector of arguments for the household problem, see list
    `args_hh_prob_names` in `input_aggregate_names.jl` and the macros `@write_args_hh_prob`
    and `@write_args_hh_prob_ss`
"""
function compute_args_hh_prob_ss(K, m_par, n_par)
    mc = 1.0
    mcw = 1.0

    # Stationary distribution of productivity
    distr_h = (n_par.Π^1000)[1, :]

    # Assumption on TFP
    Z = m_par.Z

    # Assumption on Htilde in steady state
    Htilde = n_par.Htilde

    # Assumption on q in steady state
    q = 1.0

    # Assumption on taxes
    Tlev = m_par.Tlev
    Tprog = m_par.Tprog
    Tc = m_par.Tc
    Tk = 1.0

    # Assumption
    σ = m_par.σ

    # Assumption on Hprog in steady state
    Hprog = dot(distr_h, (n_par.grid_h ./ Htilde) .^ scale_Hprog((Tprog .- 1.0), m_par))
    @assert Hprog ≈ 1.0

    # Assumption on constant labor supply
    N = m_par.N

    # Calculate interest rate using interest function
    RK_before_taxes = 1.0 .+ interest(Z, K, N, m_par)

    # Calculate capital return net of capital return tax
    RK = (RK_before_taxes .- 1.0) .* (1.0 .- (Tk .- 1.0)) .+ 1.0

    # Calculate wage
    wH = wage(Z, K, N, m_par)

    # Calculate output using output function
    Y = output(Z, K, N, m_par)

    # No-arbitrage condition for returns
    RRL = RK

    # Controls (required atm. for incomes etc)
    RRD = RRL

    Tbar = m_par.Tbar
    Π_E = eps()
    Π_U = eps()

    # Package variables for the household block, see list args_hh_prob_names
    @write_args_hh_prob()

    # Return
    return args_hh_prob
end
