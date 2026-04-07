# BASEforHANK.jl Documentation

## Introduction

This manual documents the Julia package **BASEforHANK**, which provides a toolbox for the Bayesian Solution and Estimation (BASE) of a heterogeneous-agent New-Keynesian (HANK) model.

It comes with examples that showcase how to use the package. Originally, the code accompanied the paper [Bayer, Born, and Luetticke (2024, AER)](https://www.aeaweb.org/articles?id=10.1257/aer.20201875). Note that the toolbox is not a 1-for-1 replication package for the linked paper. In particular, the preset resolution is smaller.

## First steps

### Installation

We recommend to use [Julia for VSCode IDE](https://www.julia-vscode.org) as a front-end to Julia. To get started with the toolbox, simply download or clone the folder, e.g. via `git clone` and set your `cd` to the project directory. Then start the Julia REPL and type `]` so that you can call

```julia-repl
(v1.12) pkg> activate .

(BASEtoolbox) pkg> instantiate
```

This will install all needed packages. For more on Julia environments, see [`Pkg.jl`](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project).

!!! warning
    Before you activate the environment, make sure that you are in the main directory, in which the `Project.toml` file is located. In case you accidentally activated the environment in a subfolder, empty `.toml` files will be created that you need to delete before proceeding in the correct folder.

We have tested the module on Julia Version 1.12.2 (macOS: `arm64-apple-darwin22.4.0`, Windows: `x86_64-w64-mingw32`, Linux: `x86_64-linux-gnu`). You can find out which version you are using by calling `versioninfo()` in the Julia REPL.

If you use a different editor, make sure that the environment is correctly set, as otherwise the instantiated packages might not be found.

### Toolbox folder structure

In the following, we call the root directory of the repository `BASEtoolbox.jl` (which is the directory containing, for instance, `Project.toml`). The folder structure is as follows:

`src/`: Contains the source code of the toolbox, that is:

- the main module file `BASEforHANK.jl`,
- the submodules in the folder `SubModules/`,
- and the pre-processing functions in the folder `Preprocessor/`.

`examples/`: Contains the examples that showcase the toolbox. For each example, there is a subfolder in `examples/` that contains the main file to run the example as well as all relevant files for the example. The baseline example that showcases most functions of the toolbox is given by `examples/baseline/main.jl`. This is strictly required as it serves as the baseline for testing and documentation.

`bld/`: Contains the generated files (after generating them). The folder is not part of the repository, but is created when running (certain parts of) the toolbox. That is, the folder contains:

- the generated files from the examples as subfolders of `bld/`.

`docs/`: Contains the documentation of the toolbox, that is:

- the source code in `src/`,
- and the generated documentation in `build/`.

`test/`: Contains the tests for the toolbox.

### Building the documentation

You can build the documentation *locally* by starting a new Julia REPL in the root directory of the repository, activating the environment, and running the following command: `include("docs/make.jl")`. You can access the documentation, once it is built locally, via running `python3 -m http.server --directory docs/build/`. If you then open your browser at [http://localhost:8000](http://localhost:8000), the documentation should render properly. Beyond that, the documentation is hosted via GitHub Pages and can be accessed [here](https://BASEforHANK.github.io/BASEtoolbox.jl/).

### Getting started with your model

The backbone of the toolbox is a computation algorithm to efficiently solve one- or two-asset heterogeneous agent models. The household problem, including all notation, is described in detail in [Household Problem](HouseholdProblem.md). The algorithm is described in [Computational Notes](ComputationalNotes.md).

If you want to add a new model, the recommended way is to start by copying one of the provided examples into a new folder in `examples/`. This way, you can make sure that all necessary files are present and that the toolbox can be run without any issues.

We provide a detailed description of the user inputs in a typical example in [General example structure](GeneralStructure.md). To decide which example is best suited as a starting point for your needs, you can look at the list of the provided examples in [Examples](GeneralStructure.md).

!!! tip
    If your model differs only in the aggregate model part, you can simply stay in the `examples/<your_example>/` folder. In this case you should not need to change the files `src/`. This also holds for some options built into the household problem already, e.g. taxes. Those can entirely be adjusted within your `example/<your_example>/` folder without changing the `src/` files.

!!! warning
    We recommend changing the `src/` files only to users who are willing to invest some time in understanding how our toolbox works internally. In general, changing code at one place may lead to inconsistencies, rendering the solved model invalid (if it still solves).

## Methods

The following gives an overview on the methods provided by the toolbox. They naturally build on each other. For a more detailed documentation of each method, including its main functions, see the respective section.

### Preprocessing, parsing, and incomes

Preprocessing stitches the user-provided `.mod` files together with the template functions in `src/Preprocessor/`, checks that every declared variable has a matching equation, and creates the generated functions consumed by the solver. Parsing (see [Example Structure](GeneralStructure.md)) loads `input_aggregate_names.jl`, builds the type-safe index structs, and exposes convenience macros such as `@make_struct`, `@writeXSS`, and the household-problem argument helpers. The household inputs (`input_compute_args_hh_prob_ss.jl`) are turned into executable functions via `IncomesETC` so that the steady-state solver and perturbation code can always reconstruct wages, taxes, and transfers consistent with the aggregate model.

### Steady state and preparing linearization

With model objects in place, [`compute_steadystate`](SteadyState.md) first solves the household problem (via EGM) to obtain the stationary distribution, marginal value functions, and aggregate quantities. It then calls [`prepare_linearization`](SteadyState.md#call_prepare_linearization-and-dimensionality-reduction) to build the state/control vectors, apply the first-stage DCT reduction, and assemble all bookkeeping structs (`SteadyResults`) required for perturbation, estimation, and model reduction. If you need to calibrate parameters before solving, [`run_calibration`](Calibration.md) provides an automated moment-matching loop that repeatedly invokes the steady-state solver under user-specified targets.

### Linearization and model reduction

[`linearize_full_model`](PerturbationSolution.md) differentiates the implicit equilibrium system `F(X_t, X_{t+1}) = 0`, fills Jacobian blocks with known derivatives, and solves the generalized Schur system to deliver `gx`/`hx` along with the compressed `LinearResults`. Subsequent calls to [`model_reduction`](SteadyState.md#model-reduction) construct factor representations for copula/value-function coefficients, if needed, and [`update_model`](PerturbationSolution.md#equilibrium-errors-fsys) reuses the existing Jacobians to re-linearize after parameter updates without recomputing the entire household problem. The model reduction routines build on [Bayer, Born, and Luetticke (2024)](https://www.aeaweb.org/articles?id=10.1257/aer.20201875). Models with one asset can also be solved up to second-order—the second-order solution and generalized impulse response functions are described in [PerturbationSolution](PerturbationSolution.md#solving-the-linear-system-solvediffeq) and [PostEstimation](PostEstimation.md#4-advanced-analysis-girf-and-moments)—which builds on [Bayer, Luetticke, Weiss, and Winkelmann (2025)](https://cepr.org/publications/dp19067).

### Estimation

The [`Estimation`](Estimation.md) module wraps the linear solution in a state-space model, builds the observation selector `H_sel`, and runs the Kalman filter/smoother to evaluate the likelihood. [`find_mode`](Estimation.md#mode-finding) locates the posterior mode under the supplied priors, and [`sample_posterior`](Estimation.md#bayesian-mcmc) performs random-walk Metropolis–Hastings to explore the posterior. Alternative workflows, such as [IRF Matching](IRF_matching.md), reuse the same infrastructure but swap the objective from likelihood to IRF distance.

### Plots and statistics

Post-estimation utilities (see [Post Estimation](PostEstimation.md)) compute and visualize impulse responses, variance/historical decompositions, distributional IRFs, GIRFs, and unconditional moments. The plotting helpers (`plot_irfs`, `plot_vardecomp`, `plot_hist_decomp`, …) operate on the `LinearResults`/`SteadyResults` objects and can ingest posterior draws to attach credible intervals.

For hands-on examples of the full pipeline, consult the walkthroughs in [`docs/src/examples/`](examples/baseline.md) (baseline, simplified HANK, and two-sector variants), which mirror the corresponding folders under `examples/` in the repository.
