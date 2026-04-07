# Utilities and Helpers
!!! note
    This page aggregates functionality from the `Types`, `Parsing`, and `Tools` submodules (`src/SubModules/Types/`, `src/SubModules/Parsing/`, `src/SubModules/Tools/`) and the pre-processing routines (`src/Preprocessor/`).

This page documents internal utilities, types, and helper functions used across the package. Functions are grouped by their mathematical or functional domain rather than alphabetically.

## Pre-processing
Pre-processing prepares user-supplied model inputs and steady-state specifications for the solver and linearization stages. It stitches together template files with user `.mod` content, validates equation-variable consistency, and writes generated functions used during solution and estimation.

- Purpose: Convert high-level aggregate model and steady-state definitions into executable Julia functions under `bld_example/Preprocessor/generated_fcns/`.
- Inputs: Paths and names set by the example model (e.g., `Model/input_aggregate_model.mod`, `Model/input_aggregate_names.jl`, `Model/input_aggregate_steady_state.mod`).
- Outputs: Generated files such as `FSYS_agg_generated.jl` and `prepare_linearization_generated.jl`, plus diagnostics printed to the console.

### Workflow Overview

1. Initialization and paths: Creates output folders (e.g., `bld_example/Preprocessor/generated_fcns`) and loads template files from `src/Preprocessor/template_fcns/`.
2. Aggregate model generation:
    - Reads `input_aggregate_model.mod` and injects its content into the `FSYS_agg.jl` template at the special marker.
    - Supports “replication” via a magic comment `@R` to instantiate multiple sectors/economies by symbol substitution.
    - Writes `FSYS_agg_generated.jl` with a header indicating auto-generation.
3. Consistency checks:
    - Loads variable names from `input_aggregate_names.jl`.
    - Parses `FSYS_agg_generated.jl` to identify variables that actually appear in error equations (`F[indexes.<var>]`).
    - Reports missing equations or variables and warns if the number of equations does not match expectations.
    - Ensures all household problem inputs (`args_hh_prob_names`) are included among aggregate names.
4. Steady-state preparation:
    - Reads `input_aggregate_steady_state.mod` and injects into `prepare_linearization.jl` template at its marker.
    - Sets `n_par.n_agg_eqn` from counted equations to keep solver dimensions consistent.
    - Supports the same `@R` replication logic for multi-sector steady-state definitions.
    - Writes `prepare_linearization_generated.jl` to be used by downstream linearization routines.
5. Finalization: Closes files and prints “Preprocessing Inputs… Done.” on success.

### Replication with `@R`

When `@R` is present in a `.mod` file, pre-processing will:
- Read the replication symbol and number (e.g., `@R S 3`),
- Replace occurrences of the symbol with `"", "2", "3"` across sectors/economies,
- Insert sector demarcation comments to improve readability in generated files.

### Diagnostics and Common Issues

- Missing error equations: The pre-processor checks that every aggregate variable has a corresponding model equation; otherwise it prints the missing names.
- Mismatched counts: If the number of variables does not equal the number of equations (excluding distributional names), a warning highlights inconsistencies and the offending sets.
- Household inputs: If a variable listed as input to the household problem is absent from aggregate names, pre-processing errors out.

### Generated Artifacts

- `FSYS_agg_generated.jl`: Aggregate model equations assembled from the template and user inputs.
- `prepare_linearization_generated.jl`: Steady-state and linearization preparation code, including dimension settings.

## Tools

The `Tools` submodule provides core numerical building blocks used throughout calibration, solution, and post-estimation. Below are the most crucial ones with short explanations and references to their API docs.

### Discretization and Interpolation
- `Tauchen`: Discretizes a Gaussian AR(1) process into a Markov chain with **equi-probability** bins. Bounds are picked from the Normal quantiles and transition probabilities are integrated with Gauss–Chebyshev nodes; used for income or productivity shocks.
- `ExTransition`: Alternative discretizer that keeps user-provided bounds and scales variance via `riskscale`; relies on Gauss–Chebyshev quadrature and is what the steady state/perturbation code calls when constructing income-transition matrices.
- `myinterpolate3`: Fast trilinear interpolation on structured grids; supports performance-critical lookups in HANK state spaces.

```@docs
BASEforHANK.Tools.Tauchen
BASEforHANK.Tools.ExTransition
BASEforHANK.Tools.myinterpolate3
```

### Root Finding and Solvers
- `CustomBrent`: Robust 1D root finder (Brent’s method) with package-specific defaults.
- `broyden`: Quasi-Newton method for multi-dimensional fixed-point problems; effective for equilibrium conditions.
- `Fastroot`: Optimized wrapper for vector root solving in common model setups.

```@docs
BASEforHANK.Tools.CustomBrent
BASEforHANK.Tools.broyden
BASEforHANK.Tools.Fastroot
```

### Linear Algebra Helpers
- `real_schur`: Wrapper for real Schur decomposition useful in linear solution routines.
- `stepwiseRSKron` / `stepwiseLSKron`: Structured right/left stepwise Kronecker products to accelerate large linear operations.

```@docs
BASEforHANK.Tools.real_schur
BASEforHANK.Tools.stepwiseRSKron
BASEforHANK.Tools.stepwiseLSKron
```

### Sylvester / Doubling Solvers
- `doublingGxx`: Main doubling algorithm for generalized Sylvester equations; used in perturbation and post-estimation to recover higher-order solution tensors.
- `doublingSimple`: Lightweight variant for linear Sylvester equations of the form `A*X + X*Fkron = -B`; helpful in unconditional moment calculations.

```@docs
BASEforHANK.Tools.doublingGxx
BASEforHANK.Tools.doublingSimple
```

### Probability and Transforms
- `pdf_to_cdf` / `cdf_to_pdf`: Numerical transforms between density and distribution functions; handy for distributional IRFs and diagnostics.

```@docs
BASEforHANK.Tools.pdf_to_cdf
BASEforHANK.Tools.cdf_to_pdf
```

### Derivatives and Sensitivity
- `centralderiv`: Central finite-difference derivatives for scalar/vector functions; useful when auto-diff is not applicable.

```@docs
BASEforHANK.Tools.centralderiv
BASEforHANK.Tools.centraldiff
```

### Structure ↔ Vector Mapping
- `struc_to_vec` / `vec_to_struct`: Flatten and reconstruct parameter/variable structs for optimizers and estimators.

```@docs
BASEforHANK.Tools.struc_to_vec
BASEforHANK.Tools.vec_to_struct
```

### Logging and Quiet Execution
- `quiet_call`, `@silent`, `unmute_println`, `unmute_printf`: Control logging and stdout/stderr for clean console output or benchmarking without noise.

```@docs
BASEforHANK.Tools.quiet_call
BASEforHANK.Tools.@silent
BASEforHANK.Tools.unmute_println
BASEforHANK.Tools.unmute_printf
```

## Types

The `Types` submodule provides the abstract type hierarchy used to describe models, state representations, and indexing across the package. Most structures in `Parsing/Structs` inherit some type. Using types allows us to establish interfaces and dispatch points (see [Multiple Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), an alternative to object-oriented programming in terms of structuring code, that is better suited for the Julia programming language).

- Model families: `AbstractMacroModel` with types `OneAsset`, `TwoAsset`, and `CompleteMarkets` define the degree of market incompleteness (market completeness vs uninsured idiosynratic risk vs additional trading friction). They determine many downstream choices (state dimensionality, policy/value structures, and solution routines).
- Transition of distribution: `TransitionType` with `NonLinearTransition` and `LinearTransition` marks whether using the Young/lottery-method for modelling the transition of the distribution, which is linear in optimal policies, or the DEGM-method, which allows for nonlinearities and is thus the better choice for higher-order solutions.
- Functions and matrices: Abstract containers `PolicyFunctions`, `ValueFunctions`, and `TransitionMatrices` parameterized by array types define interfaces for storing objects like grids of policy/value functions or Markov/state transition matrices.
- Distribution representation: `DistributionValues` stores the type of how the distribution is represented, including `CopulaOneAsset`, `CopulaTwoAssets`, `CDF` and `RepAgent`.
- Distribution states: `DistributionStateType` with `CDFStates` and `CopulaStates` specifies whether distributions are stored in joint, cumulative form or with Copula and marginals.
- Indexing: `Indexes`, `DistributionIndexes`, and `ValueFunctionIndexes` unify index handling for variables, equations, and grids to avoid hard-coded positions and to enable robust parsing/preprocessing.
- Transformations: `Transformations` is an abstract anchor for elements of function transformations, for example matrices of the Discrete Cosine Transformation.

These types are used throughout the methods in `Parsing`, `Preprocessor`, `PerturbationSolution`, `Estimation`, and `PostEstimation` so that code remains generic across model classes and distribution/state encodings.

## Parsing

The `Parsing` submodule translates user and example model inputs into internal, typed representations and auto-generates boilerplate code used throughout the package. It provides:

- Typed containers: parameter sets, results structs, and index structs to reference variables, shocks, distributions, and function grids without hard-coding positions.
- Name registries: `state_names`, `control_names`, `shock_names`, `aggr_names`, and their ASCII variants to ensure consistent mapping from `.mod` content to code.
- Prior/settings helpers: `prior` builds priors; `EstimationSettings` and the convenience `e_set` encapsulate estimation-specific knobs.
- Macros for codegen: generate functions and structs for indices and equations from compact specifications, keeping model authorship ergonomic.

### Workflow
1. Documentation mode: If `paths` is not defined, it automatically includes the baseline example inputs to make docs and examples self-contained.
2. Struct and index generation: `@make_struct` and `@make_struct_aggr` create index structs; `@make_fn` and `@make_fnaggr` generate functions like `produce_indexes` that map names → positions.
3. Macro utilities: `@generate_equations` expands compact model equations; argument macros like `@write_args_hh_prob*` and `@read_args_hh_prob*` manage household problem I/O signatures.
4. Priors and settings: Includes `prior.jl`, sets up `EstimationSettings`, and provides `e_set` configured with available `shock_names`.

Key macros and helpers

```@docs
BASEforHANK.Parsing.@writeXSS
BASEforHANK.Parsing.@make_fn
BASEforHANK.Parsing.@make_fnaggr
BASEforHANK.Parsing.@make_struct
BASEforHANK.Parsing.@make_struct_aggr
BASEforHANK.Parsing.@generate_equations
BASEforHANK.Parsing.@write_args_hh_prob_ss
BASEforHANK.Parsing.@write_args_hh_prob
BASEforHANK.Parsing.@read_args_hh_prob
BASEforHANK.Parsing.@read_args_hh_prob_ss
BASEforHANK.Parsing.@asset_vars
```

Additional core entries (overview):

- `ModelParameters`, `NumericalParameters`: Core parameter containers referenced across modules.
- `EstimationSettings`, `e_set`: Holds estimation control knobs (proposal scales, ME caps, horizons, etc.).
- `IndexStruct`, `IndexStructAggr`: Programmatic indices that map named objects to positions in vectors/matrices.
- `produce_indexes`, `produce_indexes_aggr`: Functions generated by macros to build index maps from name lists.
- `prior`: Constructs distributions for parameter priors consistent with `ModelParameters`.
- Name lists: `state_names`, `control_names`, `shock_names`, `aggr_names`, `distr_names` used in parsing and preprocessing.
