"""
    incomes(n_par, m_par, args_hh_prob)

Compute various types of net and gross income and the effective rate on liquid assets for
all agents in the economy, given all relevant arguments to the household problem (such as
prices, aggregates, taxes etc.), stored in the vector `args_hh_prob`.

Each element of the returned vector `net_inc`, `gross_inc`, and `RRi` is a 3D array with
dimensions nb x nk x nh, containing the net or gross income type or the effective rate for
an agent on the corresponding grid points (liquid assets, illiquid assets, productivity).

# Arguments

  - `n_par`, `m_par`: Parameters
  - `args_hh_prob`: Vector of arguments to the household problem

# Returns

  - `net_inc`:

     1. Net labor and union income for workers, adjusted for the composite good / net
        entrepreneurial profits for entrepreneurs
     2. Rental income from illiquid assets
     3. Liquid asset income
     4. Liquidation income from illiquid assets
     5. Transformation of composite to consumption for workers
     6. Like type 1, but without adjustment for the composite good

  - `gross_inc`:

     1. Gross labor and union income for workers / gross entrepreneurial profits for
        entrepreneurs
     2. Rental income from illiquid assets
     3. Liquid asset income
     4. Liquidation income from illiquid assets
     5. Gross labor income of workers, without union profits / gross entrepreneurial profits
        for entrepreneurs
  - `RRi`: Effective real gross return on liquid assets
"""
function incomes(n_par, m_par, args_hh_prob)

    # Number of income types
    net_inc_types = 6
    gross_inc_types = 5

    # Initialize arrays
    net_inc = fill(Array{typeof(args_hh_prob[1])}(undef, size(n_par.mesh_b)), net_inc_types)
    gross_inc =
        fill(Array{typeof(args_hh_prob[1])}(undef, size(n_par.mesh_b)), gross_inc_types)
    RRi = Array{typeof(args_hh_prob[1])}(undef, size(n_par.mesh_b))

    # Call in-place version
    incomes!(net_inc, gross_inc, RRi, n_par, m_par, args_hh_prob)

    return net_inc, gross_inc, RRi
end

"""
    incomes!(net_inc, gross_inc, RRi, n_par, m_par, args_hh_prob)

In-place version of [`incomes`](@ref), see that function for details.
"""
function incomes!(net_inc, gross_inc, RRi, n_par, m_par, args_hh_prob)
    @read_args_hh_prob()

    # Compute aggregate labor compensation
    labor_compensation = wH .* N ./ Hprog

    # Compute aggregate tax base of labor income tax
    tax_base = labor_compensation .+ Π_E

    #=

    The net income array consists of the following four crucial elements:
    1. Net labor and union income for workers, adjusted for the composite good / net
       entrepreneurial profits for entrepreneurs
    2. Rental income from illiquid assets
    3. Liquid asset income
    4. Liquidation income from illiquid assets

    These are "simply" the elements of the budget constaint in eq. (BC with x).

    Additionally, the net income array contains the following two elements:
    5. Transformation of composite to consumption for workers
    6. Like type 1, but without adjustment for the composite good

    The gross income array consists of the following elements:
    1. Gross labor and union income for workers / gross entrepreneurial profits for
       entrepreneurs
    2. Rental income from illiquid assets
    3. Liquid asset income
    4. Liquidation income from illiquid assets
    5. Gross labor income of workers, without union profits / gross entrepreneurial profits
       for entrepreneurs

    =#

    # Return view of last slice along last dimension
    last_slice(a) = selectdim(a, ndims(a), lastindex(a, ndims(a)))

    # Effective rate, see eq. (Return liquid)
    RRi .= RRL .* (n_par.mesh_b .> 0.0) .+ RRD .* (n_par.mesh_b .<= 0.0)

    # Type 2: gross/net income: rental income from illiquid assets
    g_rental_inc = (RK .- 1.0) ./ (1 .- (Tk .- 1.0)) .* n_par.mesh_k
    n_rental_inc = (RK .- 1.0) .* n_par.mesh_k

    # Type 3: gross/net income: liquid asset income
    liquid_asset_inc = RRi .* n_par.mesh_b

    # Type 4: gross/net income: liquidation income from illiquid assets
    liquidation_inc = q .* n_par.mesh_k

    # Type 1: gross and net labor income of workers, see eq. (Gross income) and (Tax func)
    g_labor_inc =
        labor_compensation .* (n_par.mesh_h ./ Htilde) .^ scale_Hprog((Tprog .- 1.0), m_par)
    n_labor_inc =
        labor_tax_f.(g_labor_inc, (Tlev .- 1.0), (Tprog .- 1.0), tax_base, m_par.scale_prog)

    # Type 1: net labor income of workers, adjusted for composite good, see eq. (BC with x)
    n_labor_inc_adj = scale_GHH((Tprog .- 1.0), m_par, Val(n_par.GHH)) .* n_labor_inc

    # Type 1: gross and net union profits
    g_u_profits = n_par.frac_workers .* Π_U
    n_u_profits = union_tax_f.(g_u_profits, Tbar .- 1.0)

    # Type 1: gross and net entrepreneurial profits, see eq. (Gross income) and (Tax func)
    g_e_profits = if n_par.entrepreneur
        last_slice(n_par.mesh_h) .* Π_E
    else
        ones(size(n_par.mesh_h)) .* Π_E
    end
    n_e_profits =
        labor_tax_f.(g_e_profits, (Tlev .- 1.0), (Tprog .- 1.0), tax_base, m_par.scale_prog)

    # Type 1: household-specific, non-distortionary transfers
    transfers = transfer_scheme(n_par, m_par, args_hh_prob) # vector of length nh
    # set shape: all 1s except last dimension (i.e. (1,1,nh) or (1,nh)) so that broadcasting works
    shape = ntuple(i -> i == ndims(net_inc[1]) ? length(transfers) : 1, ndims(net_inc[1]))
    transfers = reshape(transfers, shape)

    # Type 5: transformation of composite to consumption for workers
    comp_labor_GHH =
        (1.0 .- scale_GHH((Tprog .- 1.0), m_par, Val(n_par.GHH))) / (1.0 .+ (Tc .- 1.0)) .*
        n_labor_inc

    #=

    Next, combine all net and gross income sources.

    =#

    # Combine all net income sources, adjust for entrepreneurs
    net_inc[1] = n_labor_inc_adj .+ n_u_profits .+ transfers
    if n_par.entrepreneur
        last_slice(net_inc[1]) .= n_e_profits
    else
        net_inc[1] .+= n_e_profits
    end
    net_inc[2] = n_rental_inc
    net_inc[3] = liquid_asset_inc
    net_inc[4] = liquidation_inc
    net_inc[5] = comp_labor_GHH
    if n_par.entrepreneur
        last_slice(net_inc[5]) .= 0.0
    end
    net_inc[6] = n_labor_inc .+ n_u_profits .+ transfers
    if n_par.entrepreneur
        last_slice(net_inc[6]) .= n_e_profits
    else
        net_inc[6] .+= n_e_profits
    end

    # Combine all gross income sources, adjust for entrepreneurs
    gross_inc[1] = g_labor_inc .+ g_u_profits .+ transfers
    if n_par.entrepreneur
        last_slice(gross_inc[1]) .= g_e_profits
    end
    gross_inc[2] = g_rental_inc
    gross_inc[3] = liquid_asset_inc
    gross_inc[4] = liquidation_inc
    gross_inc[5] = g_labor_inc
    if n_par.entrepreneur
        last_slice(gross_inc[5]) .= g_e_profits
    else
        gross_inc[5] .+= g_e_profits
    end
end
