# Linear perturbation around steady state

!!! note
    The routines described here live mainly in the submodule `BASEforHANK.PerturbationSolution` (`src/SubModules/PerturbationSolution`) and are called from top-level helpers such as [`linearize_full_model`](@ref).

The model is linearized with respect to **aggregate variables** around the steady state. We collect all variables in period ``t`` in a single vector ``X_t`` and write all equilibrium conditions in **implicit form**

```math
F(X_t, X_{t+1}) = 0,
```

where ``X_t`` and ``X_{t+1}`` contain (expected) deviations from steady state in successive periods. Taking the total differential at the steady state ``X^*`` gives

```math
A \, \Delta X_{t+1} + B \, \Delta X_t = 0,
```

with

```math
A = \left.\frac{\partial F}{\partial X_{t+1}}\right\rvert_{X^*},
\qquad
B = \left.\frac{\partial F}{\partial X_t}\right\rvert_{X^*}.
```

We adopt the SGU convention

```math
X_t = \begin{bmatrix} d_t \\ k_t \end{bmatrix},
```

whereLinearSolution

- ``k_t`` is the vector of **state / predetermined** variables (including
  shocks),
- ``d_t`` is the vector of **control / jump** variables.

The linear solution computes matrices

- ``gx``: controls as a linear function of states,
- ``hx``: next-period states as a linear function of current states,

so that we obtain

```math
d_t = gx \, k_t, \qquad k_{t+1} = hx \, k_t,
```

with shapes ``gx \in \mathbb{R}^{n_d \times n_k}``, ``hx \in \mathbb{R}^{n_k \times n_k}`` for ``n_k`` states and ``n_d`` controls.

In the code, the equilibrium errors ``F`` are implemented as [`BASEforHANK.PerturbationSolution.Fsys`](@ref). Differentiating and solving for ``gx`` and ``hx`` is done in [`BASEforHANK.PerturbationSolution.LinearSolution`](@ref), which is called by [`linearize_full_model`](@ref) and returns the results in a `LinearResults` struct.

In the standard setting, we use the generalized Schur decomposition [^Klein] to transform the system into the linearized observation equation ``d_t = gx \, k_t`` and state transition equation ``k_{t+1} = hx \, k_t``.

---

## High-level entry point: `linearize_full_model`

```@docs
linearize_full_model
BASEforHANK.LinearSolution
```

[`linearize_full_model`](@ref) is the **user-facing** entry point for the perturbation solution. It

1. allocates empty Jacobian matrices `A`, `B` of size `sr.n_par.ntotal × sr.n_par.ntotal`,
1. calls [`BASEforHANK.LinearSolution`](@ref) to
   - check steady-state feasibility (option `ss_only = true`), or
   - compute Jacobians and solve the linear model,
2. wraps the results in a `LinearResults` struct.

The `LinearResults` object contains, among other things:

- `gx`: observation matrix mapping states to controls (`nd × nk`).
- `hx`: transition matrix mapping current states to next states (`nk × nk`).
- `A`, `B`: Jacobians of `Fsys` at steady state, with `A = ∂F/∂X'`, `B = ∂F/∂X`.
- `nk`: number of predetermined (state) variables; `nd` is inferred from the length of `X`.
- `alarm_LinearSolution`: boolean indicator if the solution encountered borderline or approximate handling (e.g. indeterminacy).

Typical usage:

```julia
sr = call_prepare_linearization(m_par)
lr = linearize_full_model(sr, m_par)
```

The resulting `lr` is then used for impulse responses, variance decompositions, higher-order derivatives, and estimation.

---

## Core linearization: `LinearSolution`

The function [`BASEforHANK.LinearSolution`](@ref) performs the actual first-order perturbation around the steady state. It executes the following main steps:

**Prepare transformations**

Construct the transformation elements for compression and decompression:

- `Γ` (shuffle matrices for marginals),
- `DC` / `IDC` (DCT forward / inverse for value functions),
- `DCD` / `IDCD` (DCT forward / inverse for copula deviations).

These define the mapping between the compressed representation used in the state vector `X` and the full-grid distributions and value functions.

**Compute Jacobians of `Fsys`**

The equilibrium error function `F` is implemented as [`BASEforHANK.PerturbationSolution.Fsys`](@ref). `LinearSolution` computes its Jacobians

```math
A = \partial F/\partial X', \qquad B = \partial F/\partial X
```

using `ForwardDiff.jacobian`, while inserting **known derivative blocks** directly [^BL]:

- contemporaneous marginal value functions → identity block in `B` over `indexes.valueFunction`,
- future marginal distributions / copula components → pre-filled via `set_known_derivatives_distr!` in `A`.

This reduces the dimension of the automatic differentiation problem and speeds up linearization substantially.

```@docs
BASEforHANK.PerturbationSolution.set_known_derivatives_distr!
```

**Solve the linear system**

Given `A` and `B`, the function calls [`BASEforHANK.PerturbationSolution.SolveDiffEq`](@ref) to compute the linear observation and state transition equations. The solver supports the options `:schur` and `:litx`; both yield consistent `gx`, `hx` mappings.

**Return solution**

`LinearSolution` returns

```julia
gx, hx, alarm_LinearSolution, nk, A, B
```

and is typically only called indirectly through [`linearize_full_model`](@ref).

Implementation details:

- Variable ordering follows an SGU-style convention internally when constructing Jacobians for FO/SO/TO; helper functions convert to the ordering used by Levintal where required for nested AD.
- Automatic differentiation uses chunking (`ForwardDiff.Chunk`) tuned for performance; steady-state feasibility is checked before differentiation.
- In verbose mode, maximum residuals per block are printed to aid diagnostics.

---

## Solving the linear system: `SolveDiffEq`

```@docs
BASEforHANK.PerturbationSolution.SolveDiffEq
```

The function [`BASEforHANK.PerturbationSolution.SolveDiffEq`](@ref) solves the generalized linear difference equation

```math
A \, \Delta X_{t+1} + B \, \Delta X_t = 0
```

to obtain `gx` and `hx`.

**Optional model reduction**

If model reduction is active, the system is pre- and post-multiplied by the reduction matrix ``\mathcal{P}`` that is computed in [`model_reduction`](@ref) and stored in `n_par.PRightAll`. This transforms the full system into a reduced one while preserving its dynamics in the directions that matter for aggregate quantities.

**Solution algorithms**

`SolveDiffEq` can use two alternative algorithms:

- `:schur` (default): generalized Schur decomposition [^Klein] to partition stable / unstable roots and select the stable solution.
- `:litx`: an algorithm based on the Implicit Function Theorem that iterates on the fixed-point conditions for the derivatives `Dg`, `Dh` [^lit].

Both return matrices that map contemporaneous states to controls (`gx`) and to future states (`hx`).

**Numerical robustness**

If `allow_approx_sol = true`, critical eigenvalues can be slightly shifted to obtain a nearby determinate solution. In such cases `alarm_LinearSolution` is set to `true` so that the caller can monitor or discard such draws in estimation.

---

## Equilibrium errors: `Fsys`

```@docs
BASEforHANK.PerturbationSolution.Fsys
```

The function [`BASEforHANK.PerturbationSolution.Fsys`](@ref) constructs and evaluates the vector of equilibrium errors

```math
F(X_t, X_{t+1})
```

for given deviations from steady state `X`, `XPrime`.

It proceeds in the following steps:

**Initialize residual vector**

Set up a vector `F` that contains the residuals for all equilibrium conditions. There are as many conditions as entries in `X` / `XPrime`. Conditions are indexed by the `IndexStruct` `indexes`.

**Generate aggregate variables**

Reconstruct all aggregate variables (for both periods) from the entries in `X` and `XPrime` using [`BASEforHANK.Parsing.@generate_equations`](@ref).

**Rebuild distributions and value functions**

Construct the full-grid marginal distributions, marginal value functions, and the copula from the steady-state values and the (compressed) deviations. For the copula, the selection of DCT coefficients that can be perturbed ensures that the perturbed function remains a copula.

**Aggregate block**

Write all equilibrium errors that depend only on **aggregate variables** into `F` using [`BASEforHANK.PerturbationSolution.Fsys_agg`](@ref).

**Backward iteration of the value function**

Compute optimal policies with [`BASEforHANK.SteadyState.EGM_policyupdate`](@ref), given future marginal value functions, prices, and individual incomes. Infer present marginal value functions from them (envelope theorem) and set the difference to the assumed present marginal value functions (in terms of their compressed deviation from steady state) as equilibrium errors.

**Forward iteration of the distribution**

Compute future marginal distributions and the copula (on the copula grid) from the previous-period distribution and optimal asset policies. Interpolate when necessary. Set the difference to the assumed future marginal distributions and copula values on the copula nodes as equilibrium errors.

**Distribution summary statistics**

Compute distribution summary statistics with [`BASEforHANK.SteadyState.distrSummaries`](@ref) and write the corresponding equilibrium conditions with their respective control variables.

**Return**

Return the residual vector `F` (and, internally, additional objects such as distributions and policy functions when requested).

Additional remarks:

- The copula is treated as the sum of two interpolants: one based on thesteady-state distribution using the full steady-state marginals as agrid, and a “deviations” function that is defined on the copula gridgenerated in `prepare_linearization`. The interpolation is carried outwith [`BASEforHANK.Tools.myinterpolate3`](@ref). By default, trilinearinterpolation is used; the code also allows for 3D Akima interpolation.
- Copula deviations are stored and manipulated in compressed DCT coefficient space and reconstructed in CDF space on the copula grid for comparisons (see `pack_distributions.jl`).
- Marginal distributions use `Γ` to respect the unit-sum constraint.

Shapes and storage conventions:

- **Copula deviations**: stored in PDF space as DCT coefficients and reconstructed in CDF space for evaluation. Grids are `nb_copula × nk_copula × nh_copula` (two-asset) or `nb_copula × nh_copula` (one-asset).
- **Marginals**: `Γ` maps `N − 1` reduced coordinates to `N` full probabilities per margin, enforcing unit sum. Income `h` is discrete and handled as a PDF; it can be converted to a CDF under nonlinear transitions.
- **Value functions**: compressed as deviations in log-inverse-marginal-utility space; uncompressed via `IDC` and exponentiated back to marginal utility with `mutil`.

---

## Model reduction

```@docs
model_reduction
BASEforHANK.PerturbationSolution.compute_reduction
```

The function [`model_reduction`](@ref) derives an approximate factor representation from a first solution of the heterogeneous-agent model [^BBL]. It stores the matrices that map **factors** to the full set of state and control variables.

Conceptually, it computes the long-run variance–covariance matrix of the state vector and constructs a projection ``\mathcal{P}`` such that the reduced variables

```math
X_r = \mathcal{P} X
```

capture most of the variation that matters for aggregate quantities. This enables fast re-solves with aggregate-only updates.

At a high level:

1. Use the linear solution (`gx`, `hx`) together with the shock processes in `m_par` to compute long-run covariances of states and controls.
2. Perform an eigenvalue decomposition and select eigenvectors associated with the dominant eigenvalues, based on thresholds in `n_par`.
3. Construct projection matrices `PRightAll`, `PRightStates` that map the original compressed coefficients to factors.
4. Store these matrices and reduced dimensions (`nstates_r`, `ncontrols_r`, `ntotal_r`) in `n_par` and return reduced index structures.

If `n_par.further_compress == false`, the model reduction step is skipped and the identity matrix is used as `PRightAll` / `PRightStates`.

---

## Reduced linear system and `update_model`

```@docs
update_model
BASEforHANK.PerturbationSolution.Fsys_agg
BASEforHANK.PerturbationSolution.LinearSolution_reduced_system
```

During estimation, many parameter updates affect only **aggregate parameters** (e.g. shock processes, policy rules, some preference parameters). The idiosyncratic block (household problem, transition matrices) often remains unchanged. To exploit this, BASEforHANK provides an efficient **re-linearization** routine.

### `LinearSolution_reduced_system`

`LinearSolution_reduced_system(sr, m_par, A, B; allow_approx_sol = false)`

1. Treats the idiosyncratic part of the Jacobians as fixed, reused from a previous full linearization.
2. Rebuilds only the aggregate part of `Fsys` and its derivatives by differentiating [`BASEforHANK.PerturbationSolution.Fsys_agg`](@ref).
3. Updates the corresponding rows / columns of `A` and `B`.
4. Calls [`BASEforHANK.PerturbationSolution.SolveDiffEq`](@ref) on the updated system to obtain new `gx`, `hx`, and `nk`.

It returns updated `gx`, `hx`, `A`, `B`, and the alarm flag.

#### Reduced residual function

Conceptually, the reduced system can be written as

```math
F_r(X_r, X'_r) = \mathcal{P}' F(\mathcal{P} X_r, \mathcal{P} X'_r),
```

where ``\mathcal{P}`` is the projection from model reduction. In this representation:

- blocks with **known-zero derivatives** (e.g. contemporaneous marginal value functions, future marginal distributions) are held constant and not differentiated,
- reduced index sets `sr.indexes_r` track which variables are treated as having nontrivial derivatives,
- `ForwardDiff` is applied only to those blocks when computing the reduced Jacobians `A_r`, `B_r`.

The reduced Jacobians are then mapped back into the full `A`, `B` by updating only the relevant rows and columns.

### `update_model`

[`update_model`](@ref) is a high-level convenience wrapper used in estimation loops:

1. Start from steady-state results `sr`, an existing linear solution `lr`, and updated parameters `m_par`.
2. Call `LinearSolution_reduced_system` with the Jacobians `lr.A`, `lr.B`.
3. Return a **new** `LinearResults` object with updated `State2Control`, `LOMstate`, `A`, `B`, `SolutionError`, and `nk`.

As long as the parameter change only affects aggregate equations, this is much faster than running `linearize_full_model` again. If parameters change the idiosyncratic structure, a new full call to [`BASEforHANK.PerturbationSolution.LinearSolution`](@ref) (and optionally `model_reduction`) is required.

---

## Higher-order solution (SO / TO)

```@docs
BASEforHANK.PerturbationSolution.compute_derivatives
BASEforHANK.PerturbationSolution.SolveSylvester
```

The higher-order solution builds on the linearization to capture local curvature (second order, SO) and optionally third-order effects (TO).

!!! warning
    Solving for the third-order terms of the Taylor expansion is not yet implemented in the toolbox.

Higher-order solutions are needed to compute welfare, aggregate risk-effects, and non-linear impulse responses.

Overview:

- **FO**: compute `A`, `B` and solve for `gx`, `hx` as described above.
- **SO**: compute the reduced Hessian `H_red` via nested automatic differentiation on the reduced system, then reconstruct the full Hessian `H` using known-zero patterns.
- **TO (optional)**: compute the reduced third-order derivatives `TO_red` similarly, then fill the full third-order tensor `TO`.

Key steps and conventions:

- **Reduced system Automatic Differentiation**: define the reduced residual function `F_r(X_r, X'_r) = P' F(P X_r, P X'_r)` and differentiate only with respect to variables that are not in constant-derivative blocks (tracked by reduced indexes). This concentrates Automatic Differentiation effort on parts of the model that move with aggregates.
- **Variable ordering**: SO / TO follow an SGU-style ordering `X = [controls′; controls; states′; states]` for Jacobian / Hessian construction; helper mappings convert to the internal ordering used to solve the higher-order equations, following Levintal (2017).

Caching and reconstruction:

- `H_red.jld2` and `TO_red.jld2` are cached to avoid recomputation during estimation loops. When present, they are loaded and used directly.
- `fill_hessian(ix_all, ix_const, length_X0, H_red)` maps the reduced Hessian back to the full `H` using index sets of variables with constant derivatives.
- `fill_thirdDeriv(ix_all, ix_const, length_X0, TO_red)` performs the analogous mapping for the third-order tensor `TO`.

Outputs:

- **FO**: `gx`, `hx`, `A`, `B`, `nk`, and `alarm_LinearSolution`.
- **SO**: additionally `H` (typically stored sparsely); dimension roughly `(length_X0 + 1) × 4 (length_X0 + 1)^2`.
- **TO (optional)**: additionally `TO` (typically stored sparsely); dimension roughly `(length_X0 + 1) × (2 (length_X0 + 1))^3`.

---

## References

[^Klein]: Paul Klein (2000), *Using the generalized Schur form to solve a multivariate linear rational expectations model*, Journal of Economic Dynamics and Control.

[^BL]: Contemporaneous marginal value functions are irrelevant for optimal individual decisions, so their effect on other model variables is zero. Due to a rich enough set of prices, the future distribution directly only affects the Fokker–Planck equation. For details, see *Solving heterogeneous agent models in discrete time with many idiosyncratic states by perturbation methods*, Quantitative Economics, Vol. 11(4), November 2020, pp. 1253–1288.

[^BBL]: [Shocks, Frictions, and Inequality in US Business Cycles](https://www.aeaweb.org/articles?id=10.1257/aer.20201875), *American Economic Review*, 2024.

[^lit]: Invoking the Implicit Function Theorem, there exist functions ``g`` and ``h`` such that $F\left(\begin{pmatrix} k \\ g(k) \end{pmatrix},\begin{pmatrix} h(k) \\ g(h(k)) \end{pmatrix}\right)=0$. Totally differentiating by ``k`` yields $B \begin{pmatrix}\mathbb{I}\\ Dg \end{pmatrix} + A \begin{pmatrix}\mathbb{I}\\ Dg \end{pmatrix} Dh = 0$. The `:lit`-type algorithms solve this equation for ``Dg`` and ``Dh`` iteratively.
