# Calibration
!!! note
    The calibration code lives in `src/SubModules/SteadyState/calibration_steadystate.jl` and is part of the `SteadyState` module.

This page documents the **calibration** workflow and how to set it up in practice. Calibration here means choosing parameters so that **model-implied moments** (e.g. capital–output ratio) are close to **target moments** from data or other sources.

The workflow is built around a single high-level function:

```julia
m_par = BASEforHANK.SteadyState.run_calibration(
    moments_function,
    cal_dict,
    m_par;
    solver = "NelderMead",
)
```

All boilerplate (paths, loading modules, etc.) is handled in the example mainboard `main_calibration.jl`. You only need to:

1. Specify **which parameters to calibrate** and **which moments to target** (`cal_dict`),
2. Write a **moments function** that computes the model moments for a given `m_par`.

---

## 1 — What Calibration Does

The calibration routine solves:

```math
\min_{\theta} \sum_{j} w_j \left( m^{\text{model}}_j(\theta) - m^{\text{target}}_j \right)^2
```

where

* $\theta$: a subset of parameters in `ModelParameters`,
* $m^{\text{target}}_j$: user-specified target moments (e.g. `K/Y = 11.22/4`),
* $m^{\text{model}}_j(\theta)$: corresponding model moments, computed from the model’s steady state,
* $w_j$: implicit weights from the optimizer.

The output is an updated `m_par` where the chosen parameters are replaced by the parameter values that best match your targets.

---

## 2 — Required Ingredients

### 2.1 `run_calibration` Signature

```julia
m_par = BASEforHANK.SteadyState.run_calibration(
    moments_function,
    cal_dict,
    m_par;
    solver = "NelderMead",
)
```

* **`moments_function::Function`**
  User-written function. Given a parameter object `m_par`, it must:

  1. Compute the steady state and required aggregates,
  2. Return a `Dict{String, Float64}` with model moments, keyed by moment names.

* **`cal_dict::Dict`**
  Collects all calibration choices (what to calibrate, what to match, and solver options).

* **`m_par::ModelParameters`**
  An instance of `::ModelParameters`. The optimizer will use this struct to define the initial parameter vector for the initialization of the optimization (depending on the optimizer) and then after running the optimization, the function will output a new set of parameters, all contained in `m_par`.

* **`solver::String`** *(optional)*
  Optimization algorithm. Currently:

  * `"NelderMead"`: local optimizer (default).
  * `"BBO"`: global optimizer using `BlackBoxOptim`.

Details on the solvers can be found within their respective packages. The default solver is `"NelderMead"`.


See [BASEforHANK.SteadyState.run_calibration](@ref) for the full signature and parameters. The tutorial below provides practical guidance that complements the API docs.

```@docs
BASEforHANK.SteadyState.run_calibration
```

---

### 2.2 The Calibration Dictionary `cal_dict`

At minimum, `cal_dict` must specify:

```julia
cal_dict = Dict()
cal_dict["target_moments"] = Dict()

# 1. Parameters to calibrate (symbols of fields in ModelParameters), for example:
cal_dict["params_to_calibrate"] = [:β, :λ]

# 2. Target moments (keyed by a descriptive string), for example:
cal_dict["target_moments"]["K/Y"] = 11.22 / 4   # Capital–output ratio
cal_dict["target_moments"]["B/K"] = 0.25       # Liquid–to–illiquid ratio

# 3. Optional: optimization options (here for Nelder–Mead)
cal_dict["opt_options"] = Optim.Options(;
    time_limit  = 1200,
    f_reltol    = 1e-3,
    store_trace = true,
    show_trace  = true,
    show_every  = 30,
)
```

**Keys:**

* `"params_to_calibrate" :: Vector{Symbol}`
  Parameters from `ModelParameters` that the optimizer may change.

* `"target_moments" :: Dict{String, Float64}`
  Mapping of **moment names** → **target values**.
  Each key **must** match the keys returned by your `moments_function`.

* `"opt_options"` *(optional)*

  * For `"NelderMead"`: `Optim.Options`.
  * For `"BBO"`: a named tuple with the optimizer settings (see Example 3 below).

---

### 2.3 The Moments Function

The core of the calibration is the function:

```julia
function my_moments_function(m_par)::Dict
    # 1. Compute steady state
    ss_full = quiet_call(call_find_steadystate, m_par)

    # 2. Compute aggregates required for your moments
    n_par = ss_full.n_par
    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()  # defines aggregates like N, etc.

    # Example aggregates
    K = ss_full.KSS
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)

    # 3. Construct model moments dictionary
    model_moments = Dict(
        "K/Y" => K / Y / 4.0,  # Quarterly capital-output ratio
    )

    return model_moments
end
```

**Requirements:**

* The function must accept **only** `m_par` as input.
* It **must** return a `Dict{String, Float64}`.
* The **keys** in this `Dict` must match those in `cal_dict["target_moments"]`.
* You are free to compute any aggregates you like, as long as they are implied by the model. We provide three examples below so you have a sense of how to precisely do this. Although you will have to find the equations to generate the aggregates you wish to match, it buys you the flexibility of being able to stipulate any moment (as long as there is a model-counterpart).

**What does it do?**

* (1) **Computes some aggregates:** `ss_full = quiet_call(call_find_steadystate, m_par)` is run. This will always need to be called, since this returns certain steady-state aggregates necessary in its own right and for the computation of other aggregates. On top, `BASEforHANK.IncomesETC.compute_args_hh_prob_ss` will always need to be called, since this returns additional aggregates possibly relevant for the user's calibration. From this, the user would be able to compute plenty moments of interest.

* (2) **Computes the model-moments:** The model-moments of interest to the user will need to be defined by in a dictionary `model_moments::Dict()`. Here in this example, you the user wanted the capital-to-output ratio and so, the user needs to find its counterpart in the code. In this case, capital in the package code is `K` and output is `Y`. The capital-to-output ratio is then computed as `K/Y`. The actual key in the dictionary, denoted as a `String`, `"K/Y"` here, can be anything and will only appear in the end, when we print the final model-moments generated from the parameters returned by the optimizer. For example, you can define it as: `model_moments = Dict("Capital-to-Output" => K / Y / 4.0)`. BUT: the keys for the moments within `model_moments` has to match the keys of `cal_dict` e.g., `cal_dict["target_moments"]["Capital-to-Output"] = K / Y / 4.0)`.

* (3) **Output:** The function in the end returns the model-moments, which the optimizer then compares to the target during the optimization procedure and ultimately, in a `PrettyTable`, once the optimizer has ran through to the specified criteria.
---

## 3 — Basic Examples

### 3.1 One-Target Calibration

Goal: calibrate β to match a target capital–output ratio.

```julia
# 1. Calibration dictionary
cal_dict_one_target = Dict()
cal_dict_one_target["target_moments"] = Dict()

cal_dict_one_target["params_to_calibrate"] = [:β]
cal_dict_one_target["target_moments"]["K/Y"] = 11.22 / 4   # target K/Y (quarterly)

cal_dict_one_target["opt_options"] = Optim.Options(;
    time_limit = 1200,
)
```

```julia
# 2. Moments function
function calculate_one_moment(m_par)
    # Steady state
    ss_full = quiet_call(call_find_steadystate, m_par)
    n_par = ss_full.n_par

    # Aggregates
    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()

    K = ss_full.KSS
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)

    model_moments = Dict("K/Y" => K / Y / 4.0)
    return model_moments
end
```

```julia
# 3. Run calibration
m_par = BASEforHANK.SteadyState.run_calibration(
    calculate_one_moment,
    cal_dict_one_target,
    m_par;
    solver = "NelderMead",
)
```

---

### 3.2 Two-Target Calibration

Goal: calibrate β and λ to match `K/Y` and `B/K`.

```julia
# 1. Calibration dictionary
cal_dict_two_targets = Dict()
cal_dict_two_targets["target_moments"] = Dict()

cal_dict_two_targets["params_to_calibrate"] = [:β, :λ]
cal_dict_two_targets["target_moments"]["K/Y"] = 11.22 / 4
cal_dict_two_targets["target_moments"]["B/K"] = 0.25

cal_dict_two_targets["opt_options"] = Optim.Options(
    time_limit = 1200,
    f_reltol   = 1e-3,
)
```

```julia
# 2. Moments function with two moments
function calculate_two_moments(m_par)
    # Steady state
    ss_full = quiet_call(call_find_steadystate, m_par)
    n_par   = ss_full.n_par

    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()

    K = ss_full.KSS
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)

    # Total bonds: integrate over steady-state distribution
    B = sum(ss_full.distrSS .* ss_full.n_par.mesh_b)

    model_moments = Dict(
        "K/Y" => K / Y / 4.0,
        "B/K" => B / K,
    )

    return model_moments
end
```

```julia
# 3. Run calibration
m_par = BASEforHANK.SteadyState.run_calibration(
    calculate_two_moments,
    cal_dict_two_targets,
    m_par;
    solver = "NelderMead",
)
```

---

## 4 — Higher-Dimensional Example and Global Optimization

When calibrating several parameters at once, one can still use `NelderMead`, but also the global optimizer such as `BlackBoxOptim` (solver `"BBO"`). Below is a compact example with five targets and five parameters.

### 4.1 Calibration Dictionary for `"BBO"`

```julia
cal_dict_BBO = Dict(
    "params_to_calibrate" => [:β, :λ, :Tlev, :ζ, :Rbar],
    "target_moments" => Dict(
        "K/Y"            => 11.22 / 4,
        "B/K"            => 0.25,
        "G/Y"            => 0.20,
        "T10W"           => 0.67,
        "Frac Borrowers" => 0.16,
    ),
    "opt_options" => (
        SearchRange = [
            (0.90, 0.999),  # β
            (0.01, 0.20),   # λ
            (1.00, 1.50),   # Tlev
            (0.00, 0.0005), # ζ
            (0.00, 0.05),   # Rbar
        ],
        Method        = :adaptive_de_rand_1_bin_radiuslimited,
        MaxTime       = 10800,   # seconds
        TraceInterval = 30,
        TraceMode     = :compact,
        TargetFitness = 1e-3,
    ),
)
```

Key points for `"BBO"`:

* `"opt_options"` must be a named tuple.
* `SearchRange` is **required**: it gives lower and upper bounds for each parameter in `"params_to_calibrate"`.
* The broader the bounds, the longer the search may take.
* More can be read about `BlackBoxOptim` here: https://github.com/robertfeldt/BlackBoxOptim.jl

### 4.2 Moments Function (Sketch)

The full example computes:

* `K/Y` (capital-output),
* `B/K` (liquid–to–illiquid ratio),
* `G/Y` (government spending-output),
* `T10W` (top 10% wealth share),
* `Frac Borrowers` (fraction of borrowers).

The structure is the same: compute the steady state, compute the necessary aggregates, then build and return a `Dict`. Conceptually:

```julia
function calculate_moments(m_par)
    ss_full = quiet_call(call_find_steadystate, m_par)
    n_par   = ss_full.n_par

    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()

    # Example aggregates (only schematic)
    K = ss_full.KSS
    B = sum(ss_full.distrSS .* ss_full.n_par.mesh_b)
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)
    # ... compute G, T, wealth distribution, TOP10Wshare, fr_borr, etc. ...

    model_moments = Dict(
        "K/Y"            => K / Y / 4.0,
        "B/K"            => B / K,
        "G/Y"            => G / Y,
        "T10W"           => TOP10Wshare,
        "Frac Borrowers" => fr_borr,
    )
    return model_moments
end
```

Then run:

```julia
m_par = BASEforHANK.SteadyState.run_calibration(
    calculate_moments,
    cal_dict_BBO,
    m_par;
    solver = "BBO",
)
```

---

## 5 — Practical Tips and Checklist

**Tips:**

* Start with **few parameters and moments** to test your setup.
* Use a **short time limit** initially (e.g. 20 minutes) and inspect how close the solution gets.
* Narrow the `SearchRange` when using `"BBO"` to reduce runtime.
* Check the optimizer trace (objective values) for convergence issues or weird behavior.

**Checklist:**

* [ ] Decide which model parameters to calibrate (`"params_to_calibrate"`).
* [ ] Decide which data (or target) moments to match and their values (`"target_moments"`).
* [ ] Write a `moments_function(m_par)` that:

  * [ ] Computes the steady state,
  * [ ] Computes aggregates needed for the target moments,
  * [ ] Returns a `Dict` with keys matching `"target_moments"`.
* [ ] Set reasonable optimization options in `"opt_options"` (and bounds for `"BBO"`).
* [ ] Run `run_calibration` with your chosen solver.
* [ ] Inspect the results: updated `m_par` and the printed model vs. target moments.

Once this is in place, calibration becomes a drop-in step in your workflow: you can quickly test alternative targets, different sets of calibrated parameters, or change solvers by modifying just `cal_dict` and the moments function.
