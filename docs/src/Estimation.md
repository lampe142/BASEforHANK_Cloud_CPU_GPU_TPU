# Estimation (likelihood-based)
!!! note
    Most functionality here lives in the submodule `Estimation` (`src/SubModules/Estimation`).

This page documents likelihood-based estimation of model parameters using the Kalman filter/smoother and (optionally) Bayesian MCMC. IRF matching is documented elsewhere and is intentionally omitted here.

## Settings

```@docs
EstimationSettings
```

The estimation settings (globally available as `e_set`) control:
- Data and observables: `data_file` (CSV path), `observed_vars_input` (observable names), `data_rename` (mapping from CSV to model names), `growth_rate_select` (transformations), and `meas_error_input` (observables with measurement error).
- Variances: `shock_names` (structural shocks whose variances are estimated) and `me_treatment` for measurement errors: `:fixed` (use data-driven caps), `:bounded` (estimate with uniform priors up to caps), or `:unbounded` (use priors in `meas_error_distr`). The cap scale is `me_std_cutoff`. See `measurement_error`.
- Numerics: `max_iter_mode` (outer iterations for mode finding), optimizer and tolerances (`optimizer`, `x_tol`, `f_tol`), and MCMC controls `ndraws`, `burnin`, `mhscale` (see `rwmh`).
- Flags: `estimate_model`, `compute_hessian` (estimate Hessian at the mode), and `multi_chain_init` (overdispersed starts for multiple chains via `multi_chain_init`).

## Data and measurement error

Measurement-error treatment and ordering are derived from `e_set` and the data. The helper builds the mapping to observable columns, prior distributions, and per-observable caps, and constructs the observation selector matrix `H_sel` (used to form `H = H_sel * [I; gx]`) that maps model variables to data.

```@docs
BASEforHANK.Estimation.measurement_error
```

## Likelihood and prior

The likelihood of a parameter vector combines the linear state-space model with the Kalman filter (or smoother) and the prior density.

- Linear solution: `LinearSolution_reduced_system` computes the reduced aggregate linearization for the current parameters and yields `gx` (state-to-control) and `hx` (state transition).
- Observation mapping: `H = H_sel * [I; gx]` where `H_sel` selects observed states/controls.
- Likelihood: `kalman_filter` on `(H, hx)` with structural/measurement covariances `SCov`, `MCov`.
- Smoother: `kalman_filter_smoother` if `smoother=true`. Missing observations in the CSV (NaN) are passed via a boolean mask to the filter/smoother.

```@docs
BASEforHANK.Estimation.likeli
BASEforHANK.Estimation.prioreval
BASEforHANK.Estimation.kalman_filter
BASEforHANK.Estimation.kalman_filter_smoother
BASEforHANK.Estimation.kalman_filter_herbst
```

## Mode finding

Find the posterior mode by maximizing the log posterior (likelihood + prior). The routine updates the reduced system between optimization rounds and can compute a Hessian at the mode.

```@docs
BASEforHANK.find_mode
BASEforHANK.mode_finding
```

## Bayesian MCMC

Random-Walk Metropolis–Hastings draws from the posterior, optionally using overdispersed initialization to seed multiple chains. The posterior for each draw is evaluated via `likeli`.

```@docs
BASEforHANK.Estimation.rwmh
BASEforHANK.Estimation.multi_chain_init
BASEforHANK.sample_posterior
```

Priors are evaluated by `BASEforHANK.Estimation.prioreval`, see above.

To obtain smoothed states at a parameter value (e.g., the posterior mean), call `likeli(...; smoother=true)` to access the Kalman smoother’s output.

## Marginal likelihood

Estimate the (log) marginal likelihood from posterior draws using the Modified Harmonic Mean estimator of Geweke (1998).

```@docs
BASEforHANK.Estimation.marginal_likeli
```

## Utilities

```@docs
BASEforHANK.Estimation.nearest_spd
```
