module PerturbationSolution

using ..Tools
using ..Parsing
using ..IncomesETC
using ..SteadyState
using ..Types

using LinearAlgebra,
    SparseArrays,
    BlockDiagonals,
    Distributions,
    Roots,
    ForwardDiff,
    Flatten,
    Setfield,
    Printf,
    PrettyTables,
    PCHIPInterpolation,
    Kronecker

using Parameters: @unpack
using MatrixEquations: lyapd

export LinearSolution,
    LinearSolution_reduced_system,
    compute_reduction,
    prepare_linearization,
    model_reduction,
    update_model,
    shuffleMatrix,
    unpack_ss_distributions,
    unpack_ss_valuefunctions,
    transformation_elements,
    unpack_perturbed_distributions,
    unpack_perturbed_valuefunctions,
    compute_derivatives,
    SolveSylvester

if !isdefined(Main, :paths)
    include("../../examples/baseline/Model/input_aggregate_names.jl")
else
    include(Main.paths["src_example"] * "/Model/input_aggregate_names.jl")
end

include("PerturbationSolution/compute_reduction.jl")
include("PerturbationSolution/FSYS.jl")
include("PerturbationSolution/LinearSolution.jl")
include("PerturbationSolution/LinearSolution_reduced_system.jl")
include("PerturbationSolution/SolveDiffEq.jl")
include("PerturbationSolution/Shuffle.jl")
include("PerturbationSolution/transformation_elements.jl")
include("PerturbationSolution/update_model.jl")
include("PerturbationSolution/model_reduction.jl")
include("PerturbationSolution/pack_distributions.jl")
include("PerturbationSolution/compute_derivatives.jl")
include("PerturbationSolution/SolveSylvester.jl")

# Documentation mode: if paths to model are not defined, the code will use the baseline example, or here, directly the template functions.
if !isdefined(Main, :paths)
    include("..//Preprocessor/template_fcns/prepare_linearization.jl")
    include("..//Preprocessor/template_fcns/FSYS_agg.jl")
else
    include(
        Main.paths["bld_example"] *
        "/Preprocessor/generated_fcns/prepare_linearization_generated.jl",
    )
    include(
        Main.paths["bld_example"] * "/Preprocessor/generated_fcns/FSYS_agg_generated.jl",
    )
end

end
