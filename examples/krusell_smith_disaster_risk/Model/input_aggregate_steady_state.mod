# Steady state values that are given already:
# - outputs of `Ksupply`, in particular: KSS, BSS
# - all variables in args_hh_prob_names
# - all variables in distr_names

## Shocks
ZSS = m_par.Z
betaSS = 1.0
deltaSS = 1 + m_par.δ_0


# Production side
YSS = output(ZSS, KSS, NSS, m_par)
ISS = m_par.δ_0 * KSS

# financial market
BDSS = max(-aggregate_asset(distrSS, :b, n_par, 0.0), eps()) # ensure non-zero debt
TotalAssetsSS = KSS

# fiscal side
RK_before_taxesSS = ((RKSS - 1.0) ./ (1.0 - (TkSS - 1.0))) + 1.0

# resource constaint
CSS = (YSS - m_par.δ_0 * KSS)
