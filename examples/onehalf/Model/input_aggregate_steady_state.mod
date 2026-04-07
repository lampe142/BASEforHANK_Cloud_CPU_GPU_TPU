@R$2 # This model has 2 replications (countries).The '$' is a placeholder.

# Steady state values that are given already:
# - outputs of `Ksupply`, in particular: KSS, BSS
# - all variables in args_hh_prob_names
# - all variables in distr_names

## Replicate outputs of KSupply for both countries
K$SS = KSS
B$SS = BSS

## Replicate all variables in args_hh_prob for both countries
wH$SS = wHSS
N$SS = NSS
Hprog$SS = HprogSS
q$SS = qSS
RRL$SS = RRLSS
RRD$SS = RRDSS
RK$SS = RKSS
Tlev$SS = TlevSS
Tprog$SS = TprogSS
Tbar$SS = TbarSS
Tc$SS = TcSS
Tk$SS = TkSS
Π_E$SS = Π_ESS
Π_U$SS = Π_USS
Htilde$SS = HtildeSS
σ$SS = σSS

# Corrections for country 2
Hprog2SS = 1.0
Tbar2SS = m_par.Tlev

## Shocks
Z$SS = m_par.Z
ZI$SS = 1.0
μ$SS = m_par.μ
μw$SS = m_par.μw
A$SS = 1.0
Rshock$SS = 1.0
Gshock$SS = 1.0
Tprogshock$SS = 1.0
Sshock$SS = 1.0

## Growth rates
Ygrowth$SS = 1.0
Bgovgrowth$SS = 1.0
Igrowth$SS = 1.0
wgrowth$SS = 1.0
Cgrowth$SS = 1.0
Tgrowth$SS = 1.0

## Further assumptions (partly also used in args_hh_prob)
mc$SS = 1.0 ./ μ$SS
mcw$SS = 1.0 ./ μw$SS
π$SS = 1.0
πw$SS = 1.0
u$SS = 1.0

πCPI$SS = 1.0
relsize = (m_par.α_S / (1.0 - m_par.α_S))
B12SS = exp(0.0)
rerSS = 1.0
p$SS = 1.0
p12SS = 1.0
derSS = 1.0
β2shockSS = 1.0
β2levelSS = 1.0
p21SS = 1.0
Trans2SS = exp(0.0)

## Variables (that are not already defined)

# nominal interest rates
RB$SS = m_par.RRB .* πCPI$SS
RL$SS = RB$SS
RD$SS = RRD$SS .* πCPI$SS

# production side
wF$SS = wage(mc$SS, Z$SS, K$SS, N$SS, m_par)
Y$SS = output(Z$SS, K$SS, N$SS, m_par)
I$SS = m_par.δ_0 * K$SS
Π_F$SS = (1.0 - mc$SS) .* Y$SS
YC$SS = Y$SS
nxSS = exp((RBSS - 1.0) * log(B12SS))

# financial market
LP$SS = RK$SS / (RB$SS / π$SS)
LPXA$SS = LP$SS
BD$SS = -aggregate_asset(distrSS, :b, n_par, 0.0)
qΠ$SS = (m_par.ωΠ .* Π_F$SS) ./ (RB$SS / πCPI$SS .- 1 .+ m_par.ιΠ) .+ 1.0
Bgov$SS = B$SS .- (qΠ$SS .- 1.0) .+ log(B12SS) .* relsize
BgovSS = BSS .- (qΠSS .- 1.0) - log(B12SS) ./ rerSS
TotalAssets$SS = B$SS + q$SS * K$SS

# fiscal side
RK_before_taxes$SS = RK$SS

# jointly determine C, T, G (interacted through consumption tax)
# resource constaint, plugged in government budget constraint and tax revenues
C$SS =
    (
        Y$SS - I$SS - (RRD$SS .- RRL$SS) * BD$SS + (RRL$SS - 1.0) * Bgov$SS - (
            (Tbar$SS .- 1.0) * (wH$SS .* N$SS + Π_E$SS + Π_U$SS) +
            (Tk$SS .- 1.0) * (RK_before_taxes$SS .- 1.0) * K$SS
        ) + log(nxSS) * relsize
    ) ./ (1.0 + (Tc$SS .- 1.0))

# Corrections for country 1
CSS =
    (
        YSS - ISS - (RRDSS .- RRLSS) * BDSS + (RRLSS - 1.0) * BgovSS - (
            (TbarSS .- 1.0) * (wHSS .* NSS + Π_ESS + Π_USS) +
            (TkSS .- 1.0) * (RK_before_taxesSS .- 1.0) * KSS
        ) - log(nxSS)
    ) ./ (1.0 + (TcSS .- 1.0))

# tax revenues
T$SS =
    (Tbar$SS .- 1.0) .* (wH$SS .* N$SS + Π_E$SS + Π_U$SS) + (Tc$SS .- 1.0) .* C$SS +
    (Tk$SS .- 1.0) .* (RK_before_taxes$SS .- 1.0) .* K$SS

# government spending from budget constraint
G$SS = T$SS - (RRL$SS - 1.0) * Bgov$SS

# remaining aggregates
BY$SS = B$SS / Y$SS
TY$SS = T$SS / Y$SS
B12YSS = exp(log(B12SS) / (4.0 * exp(YSS)))

# Lags
Ylag$SS = Y$SS
Bgovlag$SS = Bgov$SS
Tlag$SS = T$SS
Ilag$SS = I$SS
wFlag$SS = wF$SS
qlag$SS = q$SS
Clag$SS = C$SS
Tbarlag$SS = Tbar$SS
Tproglag$SS = Tprog$SS
qΠlag$SS = qΠ$SS
plag$SS = p$SS
p12lagSS = p12SS
rerlagSS = rerSS
