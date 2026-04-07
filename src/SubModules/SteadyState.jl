module SteadyState

using ..Tools
using ..Parsing
using ..IncomesETC
using ..Types

using LinearAlgebra,
    SparseArrays,
    Distributions,
    Roots,
    ForwardDiff,
    Flatten,
    Printf,
    PCHIPInterpolation,
    BlackBoxOptim,
    Optim,
    PrettyTables,
    Setfield

using Parameters: @unpack
using KrylovKit: eigsolve
using FFTW: dct, ifft

export updateW!,
    updateW,
    EGM_policyupdate!,
    EGM_policyupdate,
    DirectTransition!,
    DirectTransition,
    Ksupply,
    find_steadystate,
    first_stage_reduction,
    expected_marginal_values,
    aggregate_B_K,
    eval_cdf,
    aggregate_asset,
    aggregate_asset_helper,
    select_DCT_indices,
    distrSummaries,
    produce_distrSS,
    run_calibration

if !isdefined(Main, :paths)
    include("../../examples/baseline/Model/input_aggregate_names.jl")
else
    include(Main.paths["src_example"] * "/Model/input_aggregate_names.jl")
end

include("SteadyState/EGM/EGM_policyupdate.jl")
include("SteadyState/EGM/updateW.jl")
include("SteadyState/EGM/EvfPrime.jl")
include("SteadyState/find_steadystate.jl")
include("SteadyState/first_stage_reduction.jl")
include("SteadyState/IM_fcns/fcn_directtransition.jl")
include("SteadyState/IM_fcns/fcn_kdiff.jl")
include("SteadyState/IM_fcns/fcn_ksupply.jl")
include("SteadyState/IM_fcns/fcn_makeweights.jl")
include("SteadyState/IM_fcns/fcn_maketransition.jl")
include("SteadyState/IM_fcns/fcn_MultipleDirectTransitions.jl")
include("SteadyState/IM_fcns/fcn_aggregation.jl")
include("SteadyState/DistrSummaries.jl")
include("SteadyState/calibration_steadystate.jl")

end
