"""
Mainboard for a Krusell-Smith like one asset model, no estimation.
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
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_noestim",
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
## Calculate Steady State and prepare linearization
## ------------------------------------------------------------------------------------------

n_par_kwargs = (
    nh = 5,
    nh_coarse = 5,
    nh_copula = 5,
    transition_type = BASEforHANK.NonLinearTransition(),
    distribution_states = BASEforHANK.CDFStates(),
    transf_CDF = BASEforHANK.ParetoTransformation(),
    Kmax_coarse = 30.0,
    Kmin_coarse = 10.0,
    search_range = 0.5,
    bmax = 180.0,
);

# steady state at m_par
ss_full = call_find_steadystate(m_par; n_par_kwargs = n_par_kwargs);

# sparse DCT representation
sr_full = call_prepare_linearization(ss_full, m_par);

# compute steady state moments
K = exp.(sr_full.XSS[sr_full.indexes.KSS]);
Y = exp.(sr_full.XSS[sr_full.indexes.YSS]);
T10W = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "Capital to Output Ratio" K / Y/4.0
        "TOP 10 Wealth Share" T10W
        "Fraction of Borrower" fr_borr
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments",
    formatters = ft_printf("%.4f"),
)

## ------------------------------------------------------------------------------------------
## Solve the model using second-order perturbation
## ------------------------------------------------------------------------------------------

lr_full = linearize_full_model(sr_full, m_par);

variances = zeros(size(shock_names));
for i in eachindex(shock_names)
    variances[i] = getfield(m_par, Symbol("σ_$(shock_names[i])"))^2
end;

# set shocks except for δ to zero
@assert shock_names == [:Z, :delta, :beta]
variances[1] = 0;
variances[3] = 0;
covariances = BASEforHANK.LinearAlgebra.Diagonal(variances);

gx, hx, alarm_LinearSolution, nk, A, B, H = BASEforHANK.compute_derivatives(
    sr_full,
    m_par,
    lr_full.A,
    lr_full.B;
    buildpath = paths["bld_example"],
)
Gxx, Hxx, Gσσ, Hσσ, Xred = BASEforHANK.SolveSylvester(
    gx,
    hx,
    B,
    A,
    H,
    sr_full.n_par,
    sr_full.indexes,
    covariances,
    shock_names,
);

sor_full = BASEforHANK.SOResults(Gxx, Hxx, Gσσ, Hσσ);

jldsave(joinpath(paths["bld_example"], "sor_full.jld2"), true; sor_full);

## ------------------------------------------------------------------------------------------
## Compute moments, IRFS etc.
## ------------------------------------------------------------------------------------------

using SparseArrays

Ex, Ey, Σx, Σy = BASEforHANK.PostEstimation.uncondFirstMoment_SO_analytical(
    sr_full,
    lr_full,
    sor_full,
    covariances,
    shock_names,
);

SO_moment_K = Ey[sr_full.indexes.K - sr_full.n_par.nstates];
SO_moment_GiniW = Ey[sr_full.indexes.GiniW - sr_full.n_par.nstates];

ν = [0.0, 15 * m_par.σ_delta, 0.0];
l = 40;

# First-order IRFs
irf_fo = BASEforHANK.PostEstimation.GIRF_FO(l, ν, lr_full, sr_full);

# Second-order IRFs
xf_full_start = zeros(sr_full.n_par.nstates);
η_full = sparse(
    [getfield(sr_full.indexes, s) for s in shock_names],
    collect(1:length(variances)),
    ones(length(variances)),
    sr_full.n_par.nstates,
    length(variances),
);
irf_so_full, ~ = BASEforHANK.PostEstimation.GIRF_SO(
    l,
    ν,
    lr_full,
    lr_full,
    sr_full,
    sor_full,
    variances,
    xf_full_start,
    shock_names,
    η_full,
    η_full,
);

K_FO_irf = 100 * [irf_fo[i].y[sr_full.indexes.K - sr_full.n_par.nstates] for i = 1:l];
K_SO_irf = 100 * [irf_so_full[i].y[sr_full.indexes.K - sr_full.n_par.nstates] for i = 1:l];

GiniW_FO_irf =
    100 * [irf_fo[i].y[sr_full.indexes.GiniW - sr_full.n_par.nstates] for i = 1:l];
GiniW_SO_irf =
    100 * [irf_so_full[i].y[sr_full.indexes.GiniW - sr_full.n_par.nstates] for i = 1:l];

## ------------------------------------------------------------------------------------------
## Graphical outputs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Plotting...\n"

using Plots

p = plot(
    1:l,
    K_FO_irf;
    label = "FO IRF",
    xlabel = "Periods",
    ylabel = "K IRF (%)",
    title = "Krusell-Smith Model - First-Order IRF of K",
    legend = :topright,
    linewidth = 2,
    color = :blue,
);
plot!(p, 1:l, K_SO_irf; label = "SO IRF", linewidth = 2, color = :red);
savefig(p, joinpath(paths["bld_example"], "K_IRF_FS_model.png"));

p = plot(
    1:l,
    GiniW_FO_irf;
    label = "FO IRF",
    xlabel = "Periods",
    ylabel = "GiniW IRF (%)",
    title = "Krusell-Smith Model - First-Order IRF of GiniW",
    legend = :topright,
    linewidth = 2,
    color = :blue,
);
plot!(p, 1:l, GiniW_SO_irf; label = "SO IRF", linewidth = 2, color = :red);
savefig(p, joinpath(paths["bld_example"], "GiniW_IRF_FS_model.png"));

@printf "\n"
@printf "Done.\n"
