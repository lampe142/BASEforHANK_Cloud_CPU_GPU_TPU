# Post-Estimation Analysis
!!! note
    The code for this page lives in `src/SubModules/PostEstimation/` and is part of the `PostEstimation` submodule.

The `PostEstimation` tools help analyze model results after solving or estimating a model. This includes impulse responses (linear and distributional), variance decompositions, historical decompositions, and advanced diagnostics like GIRFs and unconditional moments.

## 1. Impulse Response Functions (IRFs)

Impulse responses trace the effect of a shock through time. In the linear case, these are computed from the state transition (`hx`) and state-to-control (`gx`) matrices derived from the linear solution.

- `compute_irfs`: Given a set of shocks and selected variables, builds the impulse responses for a specified horizon using `(State2Control, LOMstate)`. Accepts options for interval computation (via posterior draws) and returns both IRFs and optional bounds.
- `plot_irfs`: Plots a set of IRFs (possibly from multiple models) with legends, styles, and optional data overlays. Designed for quick comparison across shocks and variables.

```@docs
BASEforHANK.PostEstimation.compute_irfs
BASEforHANK.PostEstimation.plot_irfs
```

Highlight: Distributional IRFs (heatmaps) for HANK

Distributional IRFs show how the cross-sectional distribution (e.g., of assets or income) shifts over time in response to a shock. This is crucial in HANK models where aggregate dynamics depend on the distribution.

```@docs
BASEforHANK.PostEstimation.plot_distributional_irfs
```

Additional plotting helper:

`plot_irfs_cat` provides categorical plotting options to organize many IRFs by variables or shocks, useful for side-by-side comparisons and publication-ready figures.

```@docs
BASEforHANK.PostEstimation.plot_irfs_cat
```

## 2. Variance Decomposition
Forecast Error Variance Decomposition (FEVD) attributes variance in forecast errors to different structural shocks over a given horizon. These tools compute and visualize the contribution of each shock.

- `compute_vardecomp`: Uses the linear state-space representation to quantify how much each shock contributes to forecast errors of selected variables across horizons.
- `plot_vardecomp`: Visualizes FEVD results as stacked areas or bars by shock.

```@docs
BASEforHANK.PostEstimation.compute_vardecomp
BASEforHANK.PostEstimation.plot_vardecomp
```

Business-cycle-frequency variant:

- `compute_vardecomp_bcfreq`: Focuses FEVD on business-cycle frequency bands via spectral methods.
- `plot_vardecomp_bcfreq`: Plots the frequency-targeted variance contributions for clear comparison across shocks.

```@docs
BASEforHANK.PostEstimation.compute_vardecomp_bcfreq
BASEforHANK.PostEstimation.plot_vardecomp_bcfreq
```

## 3. Historical Decomposition
Historical decomposition recovers the contribution of each structural shock to the realized path of observables over a sample, consistent with the linear state-space representation.

- `compute_hist_decomp`: Reconstructs each variable as a sum of shock-specific components plus initial conditions and measurement errors.
- `plot_hist_decomp`: Plots these components over time, highlighting the shocks driving historical movements.

```@docs
BASEforHANK.PostEstimation.compute_hist_decomp
BASEforHANK.PostEstimation.plot_hist_decomp
```

## 4. Advanced Analysis (GIRF and Moments)
Generalized IRFs (GIRFs) capture potentially state-dependent or non-linear responses (e.g., sign-dependent shocks) when higher-order solutions are available. Unconditional moment tools summarize means and other moments implied by the solution.

- `GIRF_FO`, `GIRF_SO`: Compute generalized IRFs using first- or second-order approximations, allowing for asymmetric or state-dependent responses.
- `uncondFirstMoment_SO_analytical`: Provides analytical unconditional first moments under the second-order solution, useful for checks and summaries.

```@docs
BASEforHANK.PostEstimation.GIRF_FO
BASEforHANK.PostEstimation.GIRF_SO
BASEforHANK.PostEstimation.uncondFirstMoment_SO_analytical
```
