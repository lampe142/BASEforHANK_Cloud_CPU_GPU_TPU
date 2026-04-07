#=

This file contains the aggregate model equations. That is, everything but the household
planning problem which is described by one EGM backward step and one forward iteration of
the distribution.

Model equations take the form F[equation number] = (lhs) - (rhs)

Equation numbers are generated automatically and stored in the index struct. For this the
corresponding variable needs to be in the list of states or controls.

=#

# Magic comment for number (n_rep=2) of code repetitions to cover multiple economies/sectors.
# ? is the symbol to be repolaced by the number of the economy
# HANK economy is economy 1, rep agent economy is economy 2
# economy 1 does not get a number, economy 2 gets a 2, etc
# entire code gets copied n_rep times with placeholder (?) replaced by the number of the economy
@R?2

## ----------------------------------------------------------------------------------------
## Auxiliary variables
## ----------------------------------------------------------------------------------------

# Policy reaction function to Y
YREACTION? = Ygrowth?

## Taxation -------------------------------------------------------------------------------

# Mass of households in each productivity state, distribution is (nb, nk, nh)
distr_h = get_PDF_h(distr)

## Profit shares --------------------------------------------------------------------------

# Shares regarding entrepreneurs selling shares of their profits
ιΠ = (1.0 / 40.0 - 1.0 / 800.0) * m_par.shiftΠ + 1.0 / 800.0
ωΠ = ιΠ / m_par.ιΠ * m_par.ωΠ

# Slopes of the Phillips curve ------------------------------------------------------------

# Demand elasticity
η? = μ? / (μ? - 1.0)

# Implied steepness of phillips curve
κ? = η? * (m_par.κ / m_par.μ) * (m_par.μ - 1.0)

# Demand elasticity wages
ηw? = μw? / (μw? - 1.0)

# Implied steepness of wage phillips curve
κw? = ηw? * (m_par.κw / m_par.μw) * (m_par.μw - 1.0)

## Capital Utilization --------------------------------------------------------------------

# Normalize utilization to 1 in stationary equilibrium
δ_1 = exp(XSS[indexes.RK_before_taxesSS]) - 1.0 + m_par.δ_0

# Express second utilization coefficient in relative terms
δ_2 = δ_1 * m_par.δ_s

# Effective capital
Kserv? = K? * u?

# Marginal product of capital
MPKserv? = interest(mc?, Z?, Kserv?, N?, m_par) + m_par.δ_0

# Depreciation
depr? = m_par.δ_0 + δ_1 * (u? - 1.0) + δ_2 / 2.0 * (u? - 1.0)^2.0

## Open Economy Variables -----------------------------------------------------------------

# Relative size of country one
relsize = m_par.α_S / (1.0 - m_par.α_S)

# Correct discount factor country 2 (in CEE) to achieve same steady state real rate as country 1
# Interpretation: fixed liquidity preference in country 2
β2 = 1.0 ./ m_par.RRB / β2level
ω2 =
    m_par.ω * (exp.(XSS[indexes.YSS]) - exp.(XSS[indexes.GSS])) /
    (exp.(XSS[indexes.Y2SS]) - exp.(XSS[indexes.G2SS]))

## ----------------------------------------------------------------------------------------
## Aggregate equations
## ----------------------------------------------------------------------------------------

## Lagged variables -----------------------------------------------------------------------

F[indexes.Ylag?] = (log(Ylag?Prime)) - (log(Y?))
F[indexes.Bgovlag?] = (log(Bgovlag?Prime)) - (log(Bgov?))
F[indexes.Ilag?] = (log(Ilag?Prime)) - (log(I?))
F[indexes.wFlag?] = (log(wFlag?Prime)) - (log(wF?))
F[indexes.Tlag?] = (log(Tlag?Prime)) - (log(T?))
F[indexes.qlag?] = (log(qlag?Prime)) - (log(q?))
F[indexes.Clag?] = (log(Clag?Prime)) - (log(C?))
F[indexes.Tbarlag?] = (log(Tbarlag?Prime)) - (log(Tbar?))
F[indexes.Tproglag?] = (log(Tproglag?Prime)) - (log(Tprog?))
F[indexes.qΠlag?] = (log(qΠlag?Prime)) - (log(qΠ?))
F[indexes.rerlag] = log(rerlagPrime) - log(rer)
F[indexes.p12lag] = log(p12lagPrime) - log(p12)
F[indexes.plag?] = log(plag?Prime) - log(p?)

## Growth rates ---------------------------------------------------------------------------

F[indexes.Ygrowth?] = (log(Ygrowth?)) - (log(Y? / Ylag?))
F[indexes.Tgrowth?] = (log(Tgrowth?)) - (log(T? / Tlag?))
F[indexes.Bgovgrowth?] = (log(Bgovgrowth?)) - (log(Bgov? / Bgovlag?))
F[indexes.Igrowth?] = (log(Igrowth?)) - (log(I? / Ilag?))
F[indexes.wgrowth?] = (log(wgrowth?)) - (log(wF? / wFlag?))
F[indexes.Cgrowth?] = (log(Cgrowth?)) - (log(C? / Clag?))

## Fiscal policy --------------------------------------------------------------------------

# Deficit rule, see equation 33 in BBL
F[indexes.π?] =
    (log(Bgovgrowth?Prime)) - (
        -m_par.γ_B? * (log(Bgov?) - XSS[indexes.Bgov?SS]) +
        m_par.γ_Y? * log(YREACTION?) +
        m_par.γ_π? * log(π?) +
        log(Gshock?)
    )

# Rule that sets average tax rate
F[indexes.Tbar] =
    (1.0 .- m_par.spend_adj) * (log(p * G) - log(BgovPrime + T - RB / πCPI * Bgov)) +
    m_par.spend_adj * (log(Tbar .- 1.0) - log(exp(XSS[indexes.TbarSS]) - 1.0))
# This variable needs to be set for the package!

F[indexes.Tbar2] = (log(Tbar2) - XSS[indexes.Tbar2SS])

# Progressivity of labor tax, see equation 33a in BBL
F[indexes.Tprog?] =
    (log(Tprog? .- 1.0)) - (
        m_par.ρ_P * log(Tproglag? .- 1.0) +
        (1.0 - m_par.ρ_P) * (log(exp(XSS[indexes.Tprog?SS]) .- 1.0)) +
        (1.0 - m_par.ρ_P) * m_par.γ_YP * log(YREACTION?) +
        (1.0 - m_par.ρ_P) * m_par.γ_BP * (log(Bgov?) - XSS[indexes.Bgov?SS]) +
        log(Tprogshock?)
    )
# This variable needs to be set for the package!

# Level of labor tax, see equation 35 in BBL (typos!), this determines Tlev
F[indexes.Tlev] =
    (Tbar .- 1.0) - (av_labor_tax_rate(
        n_par,
        m_par,
        wH * N / Hprog,
        (Tlev .- 1.0),
        (Tprog .- 1.0),
        Π_E,
        Htilde,
        distr_h,
    ))
# This variable needs to be set for the package!

# Taxes in the rep agent country
F[indexes.Tlev2] = (Tbar2 .- 1.0) - (Tlev2 .- 1.0)

# Government budget constraint, see below equation 35 in BBL
F[indexes.G] =
    m_par.spend_adj * (log(p * G) - log(BgovPrime + T - RB / πCPI * Bgov)) +
    (1.0 .- m_par.spend_adj) * (log(G) - XSS[indexes.GSS])

F[indexes.G2] = (log(G2) - XSS[indexes.G2SS])

# Total goverment tax revenues, see below equation 35 in BBL
F[indexes.T?] =
    (log(T?)) - (log(
        (Tbar? .- 1.0) * (wH? * N? + Π_E? + Π_U?) +
        (Tc? .- 1.0) * C? +
        (Tk? .- 1.0) * (RK_before_taxes? .- 1.0) * K?,
    ))

# VAT rate (gross)
F[indexes.Tc?] = (log(Tc?)) - (XSS[indexes.Tc?SS])
# This variable needs to be set for the package!

# Capital income tax rate (gross)
F[indexes.Tk?] = (log(Tk?)) - (XSS[indexes.Tk?SS])
# This variable needs to be set for the package!

# Primary deficit shock
F[indexes.Gshock?] = (log(Gshock?Prime)) - (m_par.ρ_Gshock * log(Gshock?))

# Tax shock
F[indexes.Tprogshock?] =
    (log(Tprogshock?Prime)) - (m_par.ρ_Tprogshock * log(Tprogshock?))

F[indexes.Trans2] = ((Bgov2Prime + T2 - p2 * G2 - RB2 / πCPI2 * Bgov2) - log(Trans2))

## Monetary policy ------------------------------------------------------------------------

# Taylor rule, see equation 32 in BBL
F[indexes.RB?] =
    (log(RB?Prime)) - (
        XSS[indexes.RB?SS] +
        ((1 - m_par.ρ_R) * m_par.θ_π) * log(π?) +
        ((1 - m_par.ρ_R) * m_par.θ_Y) * log(YREACTION?) +
        m_par.ρ_R * (log(RB?) - XSS[indexes.RB?SS]) +
        log(Rshock?)
    )

# Monetary policy shock
F[indexes.Rshock?] = (log(Rshock?Prime)) - (m_par.ρ_Rshock * log(Rshock?))

## Labor market ---------------------------------------------------------------------------

# Idiosyncratic income risk (contemporaneous reaction to business cycle)
F[indexes.σ] =
    (log(σPrime)) -
    ((m_par.ρ_s * log(σ) + (1.0 - m_par.ρ_s) * m_par.Σ_n * log(Ygrowth) + log(Sshock)))
# This variable needs to be set for the package!

# Uncertainty shock
F[indexes.Sshock] = (log(SshockPrime)) - (m_par.ρ_Sshock * log(Sshock))

# Wage Phillips Curve
F[indexes.mcw?] =
    (log.(πw?) - XSS[indexes.πw?SS]) - (
        κw? * (mcw? - 1 / μw?) +
        m_par.β * (
            (log.(πw?Prime) - XSS[indexes.πw?SS]) * (N?Prime * wF?Prime) /
            (N? * wF?)
        )
    )

# Definition of real wage inflation
F[indexes.πw?] = log.(wF? / wFlag?) - log.(πw? / πCPI?)

# Process for wF-markup target
F[indexes.μw?] = (log(μw?Prime / m_par.μw)) - (m_par.ρ_μw * log(μw? / m_par.μw))

# Wages that households receive
F[indexes.wH?] = (log(wH?)) - (log(mcw? * wF?))
# This variable needs to be set for the package!

# Union profits
F[indexes.Π_U?] = (log(Π_U?)) - (log(profits_U(wF?, wH?, N?)))
# This variable needs to be set for the package!

# Labor supply
F[indexes.N?] =
    (log(N?)) - (log(
        labor_supply(
            wH?,
            Hprog?,
            (Tlev? .- 1.0),
            (Tprog? .- 1.0),
            (Tc? .- 1.0),
            m_par,
            wH? * N? / Hprog? + Π_E?,
            m_par.scale_prog,
        ),
    ))
# This variable needs to be set for the package!

# Hours-weighted average labor productivity, normalized, see equation 19b in BBL
F[indexes.Hprog] =
    (log(Hprog)) - (log(
        dot(
            distr_h[1:(end - 1)],
            (n_par.grid_h[1:(end - 1)] / Htilde) .^ scale_Hprog((Tprog .- 1.0), m_par),
        ),
    ))
# This variable needs to be set for the package!

F[indexes.Hprog2] = log.(Hprog2) - log.(1.0)

## Production -----------------------------------------------------------------------------

# Price Phillips Curve
F[indexes.mc?] =
    (log(π?) - XSS[indexes.π?SS]) - (
        κ? * (mc? - 1 / μ?) +
        m_par.β * ((log(π?Prime) - XSS[indexes.π?SS]) * Y?Prime / Y?)
    )

#'PPI Inflation Definition
F[indexes.πCPI?] = log(p? / plag?) - log((π? / πCPI?))

# Process for markup target
F[indexes.μ?] = (log(μ?Prime / m_par.μ)) - (m_par.ρ_μ * log(μ? / m_par.μ))

# Rate of return on capital
F[indexes.RK_before_taxes?] = (log(RK_before_taxes?)) - (log(1 + MPKserv? * u? - q? * depr?))
# This variable needs to be set for the package!

# Rate of return on capital, net of capital taxes
F[indexes.RK?] = (log(RK?)) - (log((RK_before_taxes? .- 1.0) * (1.0 .- (Tk? .- 1.0)) .+ 1.0))
# This variable needs to be set for the package!

# Wages that firms pay
F[indexes.wF?] = (log(wF?)) - (log(wage(mc?, Z?, Kserv?, N?, m_par)))

# Firm profits
F[indexes.Π_F?] =
    (log(Π_F?)) -
    (log(Y? * (p? - mc?) + q? * (K?Prime - (1.0 - depr?) * K?) - I?))

# Distributed profits to entrepreneurs
F[indexes.Π_E] = (log(Π_E)) - (log((1.0 - ωΠ) * Π_F + ιΠ * (qΠ - 1.0)))
# This variable needs to be set for the package!

F[indexes.Π_E2] = (log(Π_E2)) - (log((1.0 - ωΠ) * Π_F2 + ιΠ * (qΠ2 - 1.0) + log(Trans2)))

# Price of capital investment
F[indexes.q?] =
    (log(1.0)) - (log(
        ZI? *
        q? *
        (
            1.0 - m_par.ϕ / 2.0 * (Igrowth? - 1.0)^2.0 -
            m_par.ϕ * (Igrowth? - 1.0) * Igrowth?
        ) +
        m_par.β *
        ZI?Prime *
        q?Prime *
        m_par.ϕ *
        (Igrowth?Prime - 1.0) *
        (Igrowth?Prime)^2.0,
    ))
# This variable needs to be set for the package!

# Capital accumulation equation
F[indexes.I?] =
    (log(K?Prime)) - (log(
        K? * (1.0 - depr?) +
        ZI? * I? * (1.0 - m_par.ϕ / 2.0 * (Igrowth? - 1.0) .^ 2.0),
    ))

# Production function
F[indexes.Y?] = (log(Y?)) - (log(output(Z?, Kserv?, N?, m_par)))

# Output in consumption goods
F[indexes.YC?] = log(YC?) - log(p? * Y?)

# TFP
F[indexes.Z?] = (log(Z?Prime)) - (m_par.ρ_Z * log(Z?))

# Investment-good productivity
F[indexes.ZI?] = (log(ZI?Prime)) - (m_par.ρ_ZI * log(ZI?))

# Capital utilisation: optimality condition for utilization
F[indexes.u?] = (log(MPKserv?)) - (log(q? * (δ_1 + δ_2 * (u? - 1.0))))

## Asset markets --------------------------------------------------------------------------

# Discount factor shock abroad
F[indexes.β2shock] = log.(β2shockPrime) - m_par.ρ_β2shock * log.(β2shock)
F[indexes.β2level] = log.(β2levelPrime) - m_par.ρ_β2level * log.(β2level) + log.(β2shock)

# Asset pricing equation for tradable stocks
F[indexes.qΠ?] =
    (log.(RB?Prime / πCPI?Prime)) -
    (log(((qΠ?Prime - 1.0) * (1 - ιΠ) + ωΠ * Π_F?Prime) / (qΠ? - 1.0)))

# Return on liquid assets
F[indexes.RL] =
    (log(RL)) - (log(
        A * (
            (
                RB * Bgov +
                πCPI * ((qΠ - 1.0) * (1 - ιΠ) + ωΠ * Π_F + log(B12) / rer * RB2 / πCPI2)
            ) / B
        ),
    ))

F[indexes.RL2] =
    (log(RL2)) - (log(
        A2 * (
            (
                RB2 * (Bgov2 - log(B12) * relsize) +
                πCPI2 * ((qΠ2 - 1.0) * (1 - ιΠ) + ωΠ * Π_F2)
            ) / B2
        ),
    ))

# Return on liquid debt
F[indexes.RD?] = (log(RD?)) - (log(RRD? .* πCPI?))

# Total liquidity demand
F[indexes.Bgov] = (log(B)) - (log(Bgov + (qΠlag - 1.0) + log(B12) / rerlag))
F[indexes.Bgov2] =
    (log(B2Prime)) - (log(Bgov2Prime + (qΠ2 .- 1.0) - log(B12Prime) * relsize))

# Ex-post liquidity premium
F[indexes.LP?] =
    (log(LP?)) - (log((q? + RK? - 1.0) / qlag?) - log(RL? / πCPI?))

# Ex-ante liquidity premium
F[indexes.LPXA?] =
    (log(LPXA?)) -
    (log((q?Prime + RK?Prime - 1.0) / q?) - log(RL?Prime / πCPI?Prime))

# Private bond return fed-funds spread (produces goods out of nothing if negative)
F[indexes.A?] = (log(A?Prime)) - (m_par.ρ_A * log(A?))

# Real rates on liquid assets
F[indexes.RRL?] = (log(RRL?)) - (log(RL? / πCPI?))
# This variable needs to be set for the package!

# Real rates on liquid debt
F[indexes.RRD?] = (log(RRD?)) - (log(borrowing_rate_ss(RRL?, m_par)))
# This variable needs to be set for the package!

## Additional definitions -----------------------------------------------------------------

# Bond to Output ratio
F[indexes.BY?] = (log(BY?)) - (log(B? / Y?))

# Tax to output ratio
F[indexes.TY?] = (log(TY?)) - (log(T? / Y?))

# Total assets by accounting identity
F[indexes.TotalAssets?] = (log(TotalAssets?)) - (log(qlag? * K? + B?))

## Open Economy equations ------------------------------------------------------------------------

# International prices
F[indexes.p12] = log(p2) - log(p12 * rer)
F[indexes.p21] = log(p21) - log(p * rer)

# Dynamic law of one price
F[indexes.rer] = log(rer / der) - log(πCPI2 / πCPI * p12lag / plag2)

# Change of exchange rate
F[indexes.der] = log(RBPrime / RB2Prime) - log(derPrime)

#'CPI index'
F[indexes.p] =
    log(1.0) - log(
        (
            (1.0 - (1.0 - m_par.α_S) * m_par.ω) * p^(1.0 - m_par.ϵ_e) +
            (1.0 - m_par.α_S) * m_par.ω * p12^(1 - m_par.ϵ_e)
        )^(1.0 / (1.0 - m_par.ϵ_e)),
    )
F[indexes.p2] =
    log(1.0) - log(
        (
            m_par.α_S * ω2 * p21^(1.0 - m_par.ϵ_e) +
            (1.0 - m_par.α_S * ω2) * p2^(1.0 - m_par.ϵ_e)
        )^(1.0 / (1.0 - m_par.ϵ_e)),
    )

#'Home net exports'
F[indexes.nx] =
    p * (Y - G - log(nx)) - (C + I + m_par.Rbar * BD - (A - 1.0) * (RL * B / πCPI))

# Consumption as bundles of home and foreign goods
F[indexes.C] =
    log(Y - G) - log(
        p^(-m_par.ϵ_e) * (
            (1 - (1 - m_par.α_S) * m_par.ω) *
            (C + I + BD * m_par.Rbar - (A - 1.0) * (RL * B / πCPI)) +
            (1 - m_par.α_S) *
            ω2 *
            rer^(-m_par.ϵ_e) *
            (C2 + I2 + BD2 * m_par.Rbar - (A2 - 1.0) * (RL2 / πCPI2 * B2))
        ),
    )

F[indexes.C2] =
    log(Y2 - G2) - log(
        p2^(-m_par.ϵ_e) * (
            m_par.α_S *
            m_par.ω *
            rer^(m_par.ϵ_e) *
            (C + I + BD * m_par.Rbar - (A - 1.0) * (RL * B / πCPI)) +
            (1 - m_par.α_S * ω2) *
            (C2 + I2 + BD2 * m_par.Rbar - (A2 - 1.0) * (RL2 / πCPI2 * B2))
        ),
    )

## ----------------------------------------------------------------------------------------
## Closing the aggregate model, see documentation for details
## ----------------------------------------------------------------------------------------

#=

Do not delete the following lines of code!

These equations are overwritten in FSYS by the corresponding aggregation equations of
households' decisions. Here, they are simply set to close the aggregate model. This is a
trick that is exploited in the estimation when only the derivatives with respect to
aggregates is needed. These derivatives are still correct since the left-hand-side of the
equations are the same in both the purely aggregate as well as the complete model.

=#

# Scaling factor for individual productivity
F[indexes.Htilde] = (log(Htilde)) - (XSS[indexes.HtildeSS])
# This variable needs to be set for the package!

# Capital market clearing
F[indexes.K] = (log(K)) - (XSS[indexes.KSS])

# Bond market clearing
F[indexes.B] = (log(B)) - (XSS[indexes.BSS])

# IOUs
F[indexes.BD?] = (log(BD?)) - (XSS[indexes.BD?SS])
F[indexes.BD?] = (log(BD?)) - (XSS[indexes.BD?SS])

## Representative agent economy (economy 2 ...) just 3 further equations: (CEE + no-arbitrage + accumulation equation)
# fixed spread on capital returns and bond returns in 2
F[indexes.K2] = log(LPXA2) - XSS[indexes.LPXA2SS]
# consumption Euler equation for country 2
F[indexes.B2] =
    log((C2 - 1.0 ./ (1 + m_par.γ) * N2 .^ (1.0 .+ m_par.γ)) .^ (-m_par.ξ)) - log(
        Tc2 ./ Tc2Prime .* β2 .* RL2Prime ./ πCPI2Prime .*
        (C2Prime - 1.0 ./ (1.0 .+ m_par.γ) * N2Prime .^ (1.0 .+ m_par.γ)) .^ (-m_par.ξ),
    )

F[indexes.B12] =
    log(
        (1.0 - (Tbar .- 1.0)) * (wF * N + Π_E) + (p * Y - wF * N - Π_E) + A * B * RL / πCPI,
    ) - log(Tc .* C + I + BD * m_par.Rbar + BgovPrime + (qΠ .- 1.0) + log(B12Prime) / rer)

F[indexes.B12Y] = log(B12Y) - (log(B12) / (4.0 * Y))

## ----------------------------------------------------------------------------------------
## Other distributional statistics
## ----------------------------------------------------------------------------------------

#=

# TO BE EXPLAINED! CURRENTLY, THESE ARE HARD-CODED IN FSYS, FIX THIS!

=#

# other distributional statistics not used in other aggregate equations and not changing
# with parameters, but potentially with other aggregate variables are NOT included here.
# They are found in FSYS.
