using Statistics, LinearAlgebra, Dates

"""
    uncondFirstMoment_SO_analytical(sr, lr, sor, covariance; shock_names=shock_names)

Compute the unconditional first moments (mean) of states and controls
using a second-order approximation.

# Arguments

  - `sr`: steady-state results
  - `lr`: linear solution results
  - `sor`: second-order solution results
  - `covariance`: variance-covariance matrix of shocks
  - `shock_names`: names of the shocks (optional)

# Returns

  - `Ex`: unconditional mean of states
  - `Ey`: unconditional mean of controls
  - `Σx`: variance-covariance matrix of states
  - `Σy`: variance-covariance matrix of controls
"""
function uncondFirstMoment_SO_analytical(sr, lr, sor, covariance, shock_names = shock_names)
    Ns = length(shock_names)
    η = sparse(
        [getfield(sr.indexes_r, s) for s in shock_names],
        collect(1:Ns),
        ones(Ns),
        sr.n_par.nstates_r,
        Ns,
    )
    Σ_old = η * covariance * η'
    Σx = lyapd(lr.LOMstate, Σ_old) # here is the pruning: we use only first order dynamics (``x^f'') to calc variance of states
    C = 0.5 * (sor.hxx * vec(Σx) + sor.hσσ)
    Ex = doublingSimple(lr.LOMstate, 1.0, -C, eps(), 40)
    # alternative: Ex = (I-lr.LOMstate)\C
    Ey = lr.State2Control * Ex + 0.5 * (sor.gxx * vec(Σx) + sor.gσσ)
    Σy = lr.State2Control * Σx * lr.State2Control'
    return Ex, Ey, Σx, Σy
end

"""
    uncondFirstMoment_TO_analytical(sr_reduc, lr_reduc, sor_reduc, tor_reduc, Eε3, Exs, Eys, shock_names=shock_names)

Compute the unconditional first moments (mean) of states and controls
using a third-order approximation.

# Arguments

  - `sr_reduc`: reduced steady-state results
  - `lr_reduc`: reduced linear solution results
  - `sor_reduc`: reduced second-order solution results
  - `tor_reduc`: reduced third-order solution results
  - `Eε3`: expectation of triple-Kronecker power of shocks
  - `Exs`: second-order mean of states
  - `Eys`: second-order mean of controls
  - `shock_names`: names of the shocks (optional)

# Returns

  - `Ex`: unconditional mean of states
  - `Ey`: unconditional mean of controls
"""
function uncondFirstMoment_TO_analytical(
    sr_reduc,
    lr_reduc,
    sor_reduc,
    tor_reduc,
    Eε3,
    Exs,
    Eys,
    shock_names = shock_names,
)
    Ns = length(shock_names)
    η = sparse(
        [getfield(sr_reduc.indexes_r, s) for s in shock_names],
        collect(1:Ns),
        ones(Ns),
        sr_reduc.n_par.nstates_r,
        Ns,
    )
    Σ_eps = kron(η, η, η) * Eε3
    Exf3 = doublingTO(lr_reduc.LOMstate, Σ_eps, eps(), 40)
    # TEST first step: Exf3 = hx^{kron(3)} Exf3 + Σ_eps
    fst_res =
        stepwiseLSKron(
            Exf3,
            kron(lr_reduc.LOMstate, lr_reduc.LOMstate),
            lr_reduc.LOMstate,
        ) + Σ_eps - Exf3
    println("1st step residuals: ", maximum(abs.(fst_res)))
    # next step: solve for Exfxs
    c2 = stepwiseLSKron(Exf3, lr_reduc.LOMstate, 0.5 * sor_reduc.hxx)
    Exfxs = doublingSO(lr_reduc.LOMstate, c2, eps(), 40)
    # TEST second step: Exfxs = hx^{kron(2)}Exfxs + kron(hx,.5hxx)Exf3
    snd_res = stepwiseLSKron(Exfxs, lr_reduc.LOMstate, lr_reduc.LOMstate) + c2 - Exfxs
    println("2nd step residuals: ", maximum(abs.(snd_res)))
    # next step: solve for Ext
    c3 = sor_reduc.hxx * Exfxs + (tor_reduc.hxxx * Exf3 + tor_reduc.hσσσ) / 6
    Ext = doublingSimple(lr_reduc.LOMstate, 1.0, -c3, eps(), 40)
    # TEST third step: Ext = hx Ext + hxx Exfxs + (hxxx Exf3 + hσσσ)/6
    trd_res = lr_reduc.LOMstate * Ext + c3 - Ext
    println("3rd step residuals: ", maximum(abs.(trd_res)))
    Eyt =
        lr_reduc.State2Control * Ext +
        sor_reduc.gxx * Exfxs +
        (tor_reduc.gxxx * Exf3 + tor_reduc.gσσσ) / 6
    return Exs + Ext, Eys + Eyt
end

"""
    uncondSecMoment_SO_analytical(sr, lr, sor, Σx, Exs, covariance, Eϵkrϵ, Ekrϵkrϵ, shock_names=shock_names)

Compute the unconditional second moments (variance-covariance) of states and controls
using a second-order approximation.

# Arguments

  - `sr`: steady-state results
  - `lr`: linear solution results
  - `sor`: second-order solution results
  - `Σx`: variance-covariance matrix of states
  - `Exs`: second-order mean of states
  - `covariance`: variance-covariance matrix of shocks
  - `Eϵkrϵ`, `Ekrϵkrϵ`: shocks moments
  - `shock_names`: names of the shocks (optional)

# Returns

  - `Vary`: variance-covariance matrix of controls
"""
function uncondSecMoment_SO_analytical(
    sr,
    lr,
    sor,
    Σx,
    Exs,
    covariance,
    Eϵkrϵ,
    Ekrϵkrϵ,
    shock_names = shock_names,
)
    Ns = length(shock_names)
    η = sparse(
        [getfield(sr.indexes_r, s) for s in shock_names],
        collect(1:Ns),
        ones(Ns),
        sr.n_par.nstates_r,
        Ns,
    )
    Σ_old = η * covariance * η'
    ## 1st step: find E[xf^kron2 * xf^kron2']
    println("step 1")
    # Step 1.1: compute E[v_{t+1}v_{t+1}']
    timer0 = now()
    Exfkrϵxfkrϵ = kron(Σx, covariance)
    Eϵkrxfϵkrxf = kron(covariance, Σx)
    Exfkrϵϵkrxf = hcat([kron(Σx, covariance[i, :]) for i = 1:Ns]...)
    Eϵkrxfxfkrϵ = vcat([kron(Σx, covariance[i, :]') for i = 1:Ns]...)
    hxkrη = kron(lr.LOMstate, η)
    ηkrhx = kron(η, lr.LOMstate)
    ηkrη = kron(η, η)
    println("built kronecker products")
    Evv = sparse(
        hxkrη * Exfkrϵxfkrϵ * hxkrη' +
        hxkrη * Exfkrϵϵkrxf * ηkrhx' +
        ηkrhx * Eϵkrxfxfkrϵ * hxkrη' +
        ηkrhx * Eϵkrxfϵkrxf * ηkrhx' +
        ηkrη * Ekrϵkrϵ * ηkrη',
    )
    println("built Evv")
    timer_help(timer0)
    read_mem_linux()
    # Step 1.2: compute c1
    # hxkr2 = kron(lr.LOMstate,lr.LOMstate) AVOID COMPUTING THIS!
    timer05 = now()
    c1 =
        stepwiseLSKron(vec(Σx) * sparse(vec(Σ_old)'), lr.LOMstate, lr.LOMstate) +
        stepwiseRSKron(sparse(vec(Σ_old)) * vec(Σx)', lr.LOMstate', lr.LOMstate') +
        Evv
    # Step 1.3: solve Sylvester equation
    println("built c1")
    timer_help(timer05)
    read_mem_linux()
    timer1 = now()
    Exfkr2 = doublingExfkr2(lr.LOMstate, c1, eps(), 40)
    timer_help(timer1)
    read_mem_linux()
    ## 2nd step: find E[xs(xf^kron2)']
    println("step 2")
    timer2 = now()
    c2 =
        lr.LOMstate * Exs * sparse(vec(Σ_old)') +
        sor.hxx * stepwiseRSKron(Exfkr2, lr.LOMstate', lr.LOMstate') +
        sor.hxx * vec(Σx) * sparse(vec(Σ_old)') +
        0.5 *
        sor.hσσ *
        stepwiseRSKron(reshape(vec(Σx)', 1, :), lr.LOMstate', lr.LOMstate') +
        0.5 * sor.hσσ * vec(Σ_old)'
    read_mem_linux()
    Exsxfkr2 = doublingGxx(lr.LOMstate, lr.LOMstate', -c2, eps(), 40)
    read_mem_linux()
    timer_help(timer2)
    ## 3rd step: find E[xs(xs)']
    println("step 3")
    timer3 = now()
    c3 =
        lr.LOMstate * Exsxfkr2 * sor.hxx' +
        lr.LOMstate * Exs * 0.5 * sor.hσσ' +
        sor.hxx * Exsxfkr2' * lr.LOMstate' +
        sor.hxx * Exfkr2 * sor.hxx' +
        sor.hxx * vec(Σx) * 0.5 * sor.hσσ' +
        0.5 * sor.hσσ * Exs' * lr.LOMstate' +
        0.5 * sor.hσσ * vec(Σx)' * sor.hxx' +
        0.5 * sor.hσσ * 0.5 * sor.hσσ'
    read_mem_linux()
    Exs2 = doublingSimple(lr.LOMstate, lr.LOMstate', -c3, eps(), 40) # could be optimized
    read_mem_linux()
    timer_help(timer3)
    ## 4th step: find E[xf(xf^kron2)']
    println("step 4")
    timer4 = now()
    c4 = η * Eϵkrϵ * ηkrη'
    read_mem_linux()
    Exfxfkr2 = doublingGxx(lr.LOMstate, lr.LOMstate', -c4, eps(), 40)
    read_mem_linux()
    timer_help(timer4)
    ## 5th step: find E[xf(xs)']
    println("step 5")
    timer5 = now()
    c5 = lr.LOMstate * Exfxfkr2 * sor.hxx'
    read_mem_linux()
    Exfxs = doublingSimple(lr.LOMstate, lr.LOMstate', -c5, eps(), 40)
    read_mem_linux()
    timer_help(timer5)
    ## Construct Var(z)
    println("construct Varz")
    timer6 = now()
    Ezz = [
        [Σx Exfxs Exfxfkr2]
        [Exfxs' Exs2 Exsxfkr2]
        [Exfxfkr2' Exsxfkr2' Exfkr2]
    ]
    Varz =
        Ezz -
        [zeros(sr.n_par.nstates_r, 1); Exs; vec(Σx)] *
        [zeros(1, sr.n_par.nstates_r) Exs' vec(Σx)']
    read_mem_linux()
    timer_help(timer6)
    ## Construct Var(y)
    println("construct Vary")
    timer7 = now()
    C2 = [lr.State2Control lr.State2Control 0.5 * sor.gxx]
    Vary = C2 * Varz * C2'
    println("constructed Vary")
    read_mem_linux()
    timer_help(timer7)
    return Vary
end
