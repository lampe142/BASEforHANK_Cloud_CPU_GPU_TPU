"""
Mainboard for the baseline example of the BASEforHANK package, calibration.
"""

using PrettyTables, Printf;

## ------------------------------------------------------------------------------------------
## Header: set up paths, pre-process user inputs, load module
## ------------------------------------------------------------------------------------------

root_dir = replace(Base.current_project(), "Project.toml" => "");
cd(root_dir);

# set up paths for the project
paths = Dict(
    "root" => root_dir,
    "src" => joinpath(root_dir, "src"),
    "bld" => joinpath(root_dir, "bld"),
    "src_example" => @__DIR__,
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_calib",
);

# create bld directory for the current example
mkpath(paths["bld_example"]);

# pre-process user inputs for model setup
include(paths["src"] * "/Preprocessor/PreprocessInputs.jl");
include(paths["src"] * "/BASEforHANK.jl");
using .BASEforHANK;

# set BLAS threads to the number of Julia threads, prevents grabbing all
BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads());

## ------------------------------------------------------------------------------------------
## Initialize: set up model parameters
## ------------------------------------------------------------------------------------------

m_par = ModelParameters();

## ------------------------------------------------------------------------------------------
## Preparing the calibration
## ------------------------------------------------------------------------------------------

# `moments_function`
function moments_function_example(m_par)
    # calculate the steady state associated with the current parameter vector
    ss_full = quiet_call(call_find_steadystate, m_par)

    # extract numerical parameters for the calculation of the aggregates
    n_par = ss_full.n_par

    # Compute aggregates
    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()

    # Compute aggregates
    # capital
    K = ss_full.KSS

    # bonds
    B = sum(ss_full.distrSS .* ss_full.n_par.mesh_b)

    # marginal cost, wages, output
    mc = 1.0 ./ m_par.μ
    wF = BASEforHANK.IncomesETC.wage(mc, m_par.Z, K, N, m_par)
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)

    # taxes
    T =
        (Tbar - 1.0) .* (1.0 ./ m_par.μw .* wF .* N) + # labor income
        (Tbar - 1.0) .* Π_E + # profit income
        (Tbar - 1.0) * ((1.0 .- 1.0 ./ m_par.μw) .* wF .* N) # union profit income

    # government bonds
    RRL = m_par.RRB # in ss
    qΠ = m_par.ωΠ .* (1.0 .- 1.0 ./ m_par.μ) .* Y ./ (RRL .- 1 .+ m_par.ιΠ) + 1.0
    Bgov = B .- qΠ .+ 1.0

    # Bgov = B - (m_par.ωΠ .* (1.0 .- 1.0 ./ m_par.μ) .* Y ./ (RRB .- 1 .+ m_par.ιΠ)) # Needs to be checked.

    # government spending
    G = T - (RRL - 1.0) * Bgov

    # calculate the fraction of borrowers
    distrSS = ss_full.distrSS
    fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0)

    # Price of capital is 1 in the steady-state
    q = 1

    # calculate the Top 10% wealth share
    total_wealth = Array{eltype(distrSS)}(undef, n_par.nk .* n_par.nb)
    for k = 1:(n_par.nk)
        for b = 1:(n_par.nb)
            total_wealth[b + (k - 1) * n_par.nb] = n_par.grid_b[b] .+ q .* n_par.grid_k[k]
        end
    end
    # Wealth shares
    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]
    total_wealth_pdf = sum(distrSS; dims = 3)
    total_wealth_pdf = total_wealth_pdf[IX]
    total_wealth_cdf = cumsum(total_wealth_pdf)
    total_wealth_w = total_wealth .* total_wealth_pdf # weighted
    wealthshares = cumsum(total_wealth_w) ./ sum(total_wealth_w)
    TOP10Wshare =
        1.0 -
        BASEforHANK.Tools.mylinearinterpolate(total_wealth_cdf, wealthshares, [0.9])[1]

    # Compute model moments -- note that the keys match the keys of `target_moments`
    model_moments = Dict(
        "K/Y" => K / Y / 4,
        "B/K" => B / K,
        "G/Y" => G / Y,
        "T10W" => TOP10Wshare,
        "Frac Borrowers" => fr_borr,
    )

    return model_moments
end;

# Generate dictionary for calibration
using Optim;

# For Nelder-Mead
cal_dict = Dict(
    "params_to_calibrate" => [:β, :λ, :Tlev, :ζ, :Rbar],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 11.22 / 4,  # Capital-output ratio
        "B/K" => 0.25,  # Liquid to illiquid ratio
        "G/Y" => 0.20,  # Gov. spending-output ratio
        "T10W" => 0.67,  # Top 10% wealth share
        "Frac Borrowers" => 0.16,  # Fraction of borrowers
    ),
    # One must change options for their respective setting!
    "opt_options" => Optim.Options(;
        time_limit = 10800,
        show_trace = true,
        show_every = 10, # iteration count
        f_reltol = 1e-3,   # stops if fitness ≤ tolerance
    ),
);

# Run calibration. Exports parameters
m_par = BASEforHANK.SteadyState.run_calibration(
    moments_function_example,
    cal_dict,
    m_par;
    solver = "NelderMead",
);

## ------------------------------------------------------------------------------------------
## Calculate Steady State and prepare linearization
## ------------------------------------------------------------------------------------------

# steady state at m_par
ss_full = call_find_steadystate(m_par);

# sparse DCT representation
sr_full = call_prepare_linearization(ss_full, m_par);

# compute steady state moments
K = exp.(sr_full.XSS[sr_full.indexes.KSS]);
B = exp.(sr_full.XSS[sr_full.indexes.BSS]);
Bgov = exp.(sr_full.XSS[sr_full.indexes.BgovSS]);
Y = exp.(sr_full.XSS[sr_full.indexes.YSS]);
T10W = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS]);
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "Liquid to Illiquid Assets Ratio" B/K
        "Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y
        "TOP 10 Wealth Share" T10W
        "Fraction of Borrower" fr_borr
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments",
    formatters = ft_printf("%.4f"),
)

## ------------------------------------------------------------------------------------------
## Linearize the full model, find sparse state-space representation
## ------------------------------------------------------------------------------------------

lr_full = linearize_full_model(sr_full, m_par);

## ------------------------------------------------------------------------------------------
## Compute all IRFs, VDs, and BCVDs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Compute IRFs, VDs, and BCVDs...\n"

# Get indices of the shocks
exovars = [getfield(sr_full.indexes, shock_names[i]) for i = 1:length(shock_names)];

# Get standard deviations of the shocks
stds = [getfield(sr_full.m_par, Symbol("σ_", i)) for i in shock_names];

# Compute IRFs
IRFs, _, IRFs_order, IRFs_dist = compute_irfs(
    exovars,
    lr_full.State2Control,
    lr_full.LOMstate,
    sr_full.XSS,
    sr_full.indexes;
    init_val = stds,
    distribution = true,
    comp_ids = sr_full.compressionIndexes,
    n_par = sr_full.n_par,
);

# Compute variance decomposition of IRFs
VDs = compute_vardecomp(IRFs);

# Compute business cycle frequency variance decomposition
VDbcs, UnconditionalVar =
    compute_vardecomp_bcfreq(exovars, stds, lr_full.State2Control, lr_full.LOMstate);

## ------------------------------------------------------------------------------------------
## Graphical outputs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Plotting...\n"

mkpath(paths["bld_example"] * "/IRFs");
plot_irfs(
    [
        (:Z, "TFP"),
        (:ZI, "Inv.-spec. tech."),
        (:μ, "Price markup"),
        (:μw, "Wage markup"),
        (:A, "Risk premium"),
        (:Rshock, "Mon. policy"),
        (:Gshock, "Structural deficit"),
        (:Tprogshock, "Tax progr."),
        (:Sshock, "Income risk"),
    ],
    [
        (:Ygrowth, "Output growth"),
        (:Cgrowth, "Consumption growth"),
        (:Igrowth, "Investment growth"),
        (:N, "Employment"),
        (:wgrowth, "Wage growth"),
        (:RB, "Nominal rate"),
        (:π, "Inflation"),
        (:σ, "Income risk"),
        (:Tprog, "Tax progressivity"),
        (:TOP10Wshare, "Top 10 wealth share"),
        (:TOP10Ishare, "Top 10 inc. share"),
    ],
    [(IRFs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
);

@printf "\n"
@printf "Done.\n"
