#=

This file contains the aggregate model equations. That is, everything but the household
planning problem which is described by one EGM backward step and one forward iteration of
the distribution.

Model equations take the form F[equation number] = (lhs) - (rhs)

Equation numbers are generated automatically and stored in the index struct. For this the
corresponding variable needs to be in the list of states or controls.

=#

## ----------------------------------------------------------------------------------------
## Auxiliary variables
## ----------------------------------------------------------------------------------------


## ----------------------------------------------------------------------------------------
## Aggregate equations
## ----------------------------------------------------------------------------------------

## Fiscal policy --------------------------------------------------------------------------

# Average tax rate, see equation 34 in BBL (here simplified)
F[indexes.Tbar] = (log(Tbar)) - (0.0)
# This variable needs to be set for the package!

# Progressivity of labor tax, see equation 33a in BBL (here simplified)
F[indexes.Tprog] = (log(Tprog)) - (0.0)
# This variable needs to be set for the package!

# Level of labor tax, see equation 35 in BBL (typos!), this determines Tlev
F[indexes.Tlev] = (log(Tlev)) - (0.0)
# This variable needs to be set for the package!

# VAT rate (gross)
F[indexes.Tc] = (log(Tc)) - (0.0)
# This variable needs to be set for the package!

# Capital income tax rate (gross)
F[indexes.Tk] = (log(Tk)) - (0.0)
# This variable needs to be set for the package!

## Labor market ---------------------------------------------------------------------------

# Idiosyncratic income risk (contemporaneous reaction to business cycle)
F[indexes.σ] = (log(σ)) - (0.0)
# This variable needs to be set for the package!

# Wages that households receive
F[indexes.wH] = (log(wH)) - (log(wage(Z, K, N, m_par)))
# This variable needs to be set for the package!

# Union profits
F[indexes.Π_U] = (log(Π_U)) - (XSS[indexes.Π_USS])
# This variable needs to be set for the package!

# Labor supply
F[indexes.N] =
    (log(N)) - (XSS[indexes.NSS])
# This variable needs to be set for the package!

# Hours-weighted average labor productivity, normalized, see equation 19b in BBL
F[indexes.Hprog] = (log(Hprog)) - (XSS[indexes.HprogSS])
# This variable needs to be set for the package!

## Production -----------------------------------------------------------------------------

# Rate of return on capital
F[indexes.RK_before_taxes] =
    (log(RK_before_taxes)) - (log(1 + interest(Z, K, N, m_par; delta = delta - 1.0)))
# This variable needs to be set for the package!

# Rate of return on capital, net of capital taxes
F[indexes.RK] = (log(RK)) - (log((RK_before_taxes .- 1.0) * (1.0 .- (Tk .- 1.0)) .+ 1.0))
# This variable needs to be set for the package!

# Distributed profits to entrepreneurs
F[indexes.Π_E] = (log(Π_E)) - XSS[indexes.Π_ESS]
# This variable needs to be set for the package!

# Price of capital investment
F[indexes.q] = (log(q)) - XSS[indexes.qSS]
# This variable needs to be set for the package!

# Capital accumulation equation
F[indexes.I] = (log(KPrime)) - (log(K * (1.0 .- (delta .- 1.0)) + I))

# Production function
F[indexes.Y] = (log(Y)) - (log(output(Z, K, N, m_par)))

# TFP
F[indexes.Z] = (log(ZPrime)) - (m_par.ρ_Z * log(Z))

# Capital destruction
F[indexes.delta] =
    log.(deltaPrime) - m_par.ρ_delta * log.(delta) -
    (1 - m_par.ρ_delta) * XSS[indexes.deltaSS]

# Time preference shock
F[indexes.beta] = log.(betaPrime) - m_par.ρ_beta * log.(beta)

## Asset markets --------------------------------------------------------------------------

# Real rates on liquid assets
F[indexes.RRL] = (log(RRL)) - (log(RK))
# This variable needs to be set for the package!

# Real rates on liquid debt
F[indexes.RRD] = (log(RRD)) - (log(RK))
# This variable needs to be set for the package!

## Market clearing ------------------------------------------------------------------------

# Resource constraint
F[indexes.C] = (log(Y - I)) - (log(C))

## Additional definitions -----------------------------------------------------------------

# Capital by accounting identity
F[indexes.K] = (log(K)) - (log(TotalAssets)) # qlag * K = TotalAssets (but q = 1 always)

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

# Total assets market clearing
F[indexes.TotalAssets] = (log(TotalAssets)) - (XSS[indexes.TotalAssetsSS])

# IOUs
F[indexes.BD] = (log(BD)) - (XSS[indexes.BDSS])

## ----------------------------------------------------------------------------------------
## Other distributional statistics
## ----------------------------------------------------------------------------------------

#=

# TO BE EXPLAINED! CURRENTLY, THESE ARE HARD-CODED IN FSYS, FIX THIS!

=#

# other distributional statistics not used in other aggregate equations and not changing
# with parameters, but potentially with other aggregate variables are NOT included here.
# They are found in FSYS.
