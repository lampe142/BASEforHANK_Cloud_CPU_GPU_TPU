"""
    @generate_equations()

Write out the expansions around steady state for all variables in `aggr_names`, i.e.
generate code that reads aggregate states/controls from steady state deviations.

Equations take the form of (with variable `r` as example):

  - `r       = exp.(XSS[indexes.rSS] .+ X[indexes.r])`
  - `rPrime  = exp.(XSS[indexes.rSS] .+ XPrime[indexes.r])`

# Requires

(module) global `aggr_names`
"""
macro generate_equations()
    ex = quote end
    for j in aggr_names
        i = Symbol(j)
        varnamePrime = Symbol(i, "Prime")
        varnameSS = Symbol(i, "SS")
        ex_aux = quote
            $i = exp.(XSS[indexes.$varnameSS] .+ X[indexes.$i])
            $varnamePrime = exp.(XSS[indexes.$varnameSS] .+ XPrime[indexes.$i])
        end
        append!(ex.args, ex_aux.args)
    end
    return esc(ex)
end

"""
    @write_args_hh_prob()

Write all variables defined in the global string vector `args_hh_prob_names` into a vector
`args_hh_prob` based on the variables with the according names in local scope. The type of
the vector is inferred from the first variable in `args_hh_prob_names`.

# Requires

(module) global `args_hh_prob_names`
"""
macro write_args_hh_prob()
    varname = Symbol(args_hh_prob_names[1])
    ex = quote
        args_hh_prob = Vector{typeof($varname)}(undef, length(args_hh_prob_names))
    end
    for (i, j) in enumerate(args_hh_prob_names)
        varname = Symbol(j)
        ex_aux = quote
            args_hh_prob[$i] = $varname
        end
        append!(ex.args, ex_aux.args)
    end
    return esc(ex)
end

"""
    @write_args_hh_prob_ss()

See [`@write_args_hh_prob`](@ref), however, for the case where the variables are called with
a suffix "SS".

# Requires

(module) global `args_hh_prob_names`
"""
macro write_args_hh_prob_ss()
    varname = Symbol(args_hh_prob_names[1], "SS")
    ex = quote
        args_hh_prob = Vector{typeof($varname)}(undef, length(args_hh_prob_names))
    end
    for (i, j) in enumerate(args_hh_prob_names)
        varname = Symbol(j, "SS")
        ex_aux = quote
            args_hh_prob[$i] = $varname
        end
        append!(ex.args, ex_aux.args)
    end
    return esc(ex)
end

"""
    @read_args_hh_prob()

Read all variables defined in the global string vector `args_hh_prob_names` into local scope
based on the variables with the according names in the vector `args_hh_prob`.

# Requires

(module) global `args_hh_prob_names`, `args_hh_prob`
"""
macro read_args_hh_prob()
    ex = quote end
    for (i, j) in enumerate(args_hh_prob_names)
        varname = Symbol(j)
        ex_aux = quote
            $varname = args_hh_prob[$i]
        end
        append!(ex.args, ex_aux.args)
    end
    return esc(ex)
end

"""
    @read_args_hh_prob_ss()

See [`@read_args_hh_prob`](@ref), however, for the case where the variables are called with
a suffix "SS".

# Requires

(module) global `args_hh_prob_names`, `args_hh_prob`
"""
macro read_args_hh_prob_ss()
    ex = quote end
    for (i, j) in enumerate(args_hh_prob_names)
        varname = Symbol(j, "SS")
        ex_aux = quote
            $varname = args_hh_prob[$i]
        end
        append!(ex.args, ex_aux.args)
    end
    return esc(ex)
end

"""
    @writeXSS()

Write all single steady state variables into vectors XSS / XSSaggr.

# Requires

(module) globals `state_names`, `control_names`, `aggr_names`
"""
macro writeXSS()
    ex = quote
        XSS = [distrXSS_vec[:];]
    end
    for j in state_names
        varnameSS = Symbol(j, "SS")
        ex_aux = quote
            append!(XSS, log($varnameSS))
        end
        append!(ex.args, ex_aux.args)
    end

    ex_aux = quote
        append!(XSS, [vfSS_vec[:];])
    end
    append!(ex.args, ex_aux.args)
    for j in control_names
        varnameSS = Symbol(j, "SS")
        ex_aux = quote
            append!(XSS, log($varnameSS))
        end
        append!(ex.args, ex_aux.args)
    end
    ex_aux = quote
        XSSaggr = [0.0]
    end
    append!(ex.args, ex_aux.args)
    for j in aggr_names
        varnameSS = Symbol(j, "SS")
        ex_aux = quote
            append!(XSSaggr, log($varnameSS))
        end
        append!(ex.args, ex_aux.args)
    end
    ex_aux = quote
        deleteat!(XSSaggr, 1)
    end
    append!(ex.args, ex_aux.args)

    return esc(ex)
end

"""
    @make_fnaggr(fn_name)

Create function `fn_name` that returns an instance of `IndexStructAggr` (created by
[`@make_struct_aggr`](@ref)), mapping aggregate states and controls to values `1` to
`length(aggr_names)` (both steady state and deviation from it).

# Requires

(module) global `aggr_names`
"""
macro make_fnaggr(fn_name)
    n_states = length(aggr_names)
    fieldsSS_states = [:($i) for i = 1:n_states]
    fields_states = [:($i) for i = 1:n_states]
    esc(quote
        function $(fn_name)(n_par)
            indexes = IndexStructAggr($(fieldsSS_states...), $(fields_states...))
            return indexes
        end
    end)
end

"""
    @make_fn(fn_name)

Create function `fn_name` that returns an instance of `IndexStruct` (created by
[`@make_struct`](@ref)), mapping states and controls to indexes inferred from numerical
parameters and compression indexes.

# Requires

(module) global `state_names`, `control_names`
"""
macro make_fn(fn_name)
    n_states = length(state_names)
    n_controls = length(control_names)
    fieldsSS_states = [:(nIdxDistrSS + $i) for i = 1:n_states]
    fields_states = [:(nIdxDistr + $i) for i = 1:n_states]
    fieldsSS_controls = [
        :(nIdxDistrSS + n_states + sum(nIdxValuesSS; init = 0) + $i) for i in (1:n_controls)
    ]
    fields_controls =
        [:(nIdxDistr + n_states + sum(nIdxValues; init = 0) + $i) for i in (1:n_controls)]
    esc(
        quote
            function $(fn_name)(n_par, compressionIndexesVf, compressionIndexesD)
                nb = n_par.nb
                nk = typeof(n_par.model) == TwoAsset ? n_par.nk : 1
                nh = n_par.nh
                n_states = $(n_states)
                nIdxValuesSS = [nb * nh * nk for _ = 1:length(compressionIndexesVf)]
                nIdxValues = length.(compressionIndexesVf)

                # Using multiple dispatch functions for index creation
                distrIdxSS = create_distribution_indexes(
                    n_par,
                    length(compressionIndexesD),
                    n_par.model,
                    n_par.distribution_states;
                    ss = true,
                )
                nIdxDistrSS = sum(length.(struc_to_vec(distrIdxSS)); init = 0) # Count total number of indexes
                distrIdx = create_distribution_indexes(
                    n_par,
                    length(compressionIndexesD),
                    n_par.model,
                    n_par.distribution_states;
                    ss = false,
                )
                nIdxDistr = sum(length.(struc_to_vec(distrIdx)); init = 0) # Count total number of indexes
                vfIdxSS = create_value_function_indexes(
                    n_par.model,
                    n_states,
                    nIdxDistrSS,
                    nIdxValuesSS,
                )
                vfIdx = create_value_function_indexes(
                    n_par.model,
                    n_states,
                    nIdxDistr,
                    nIdxValues,
                )

                indexes = IndexStruct(
                    distrIdxSS, # Distribution(s)
                    $(fieldsSS_states...),
                    vfIdxSS, # Value functions
                    $(fieldsSS_controls...),
                    distrIdx, # Distribution(s)
                    $(fields_states...),
                    vfIdx, # Value functions
                    $(fields_controls...),
                )
                return indexes
            end
        end,
    )
end

# Multiple dispatch functions for creating index structures
"""
    create_distribution_indexes(model::OneAsset, nb, nk, nh, nIdxDistr)

Create distribution indexes for OneAsset model where CDF is the state for the distribution.
"""
function create_distribution_indexes(
    n_par,
    nIdxDistr::Int,
    model::Union{OneAsset,TwoAsset},
    distribution_states::CDFStates;
    ss = false,
)
    ndistr = ss ? nIdxDistr : nIdxDistr - 1
    return CDFIndexes(1:ndistr)
end

"""
    create_distribution_indexes(model::OneAsset, nb, nk, nh, nIdxDistr)

Create distribution indexes for OneAsset model where Copula and marginals are states for the distribution.
"""
function create_distribution_indexes(
    n_par,
    nIdxDistr::Int,
    model::OneAsset,
    distribution_states::CopulaStates;
    ss = false,
)
    nb = ss ? n_par.nb : n_par.nb - 1
    nh = ss ? n_par.nh : n_par.nh - 1
    nCop = ss ? n_par.nb * n_par.nh : nIdxDistr
    return CopulaOneAssetIndexes(
        1:nCop, # Copula
        nCop .+ (1:nb), # distr_b
        nCop .+ (nb) .+ (1:nh), # distr_h
    )
end

"""
    create_distribution_indexes(model::TwoAsset, nb, nk, nh, nIdxDistr)

Create distribution indexes for TwoAsset model nIdxCOP.
"""
function create_distribution_indexes(
    n_par,
    nIdxDistr::Int,
    model::TwoAsset,
    distribution_states::CopulaStates;
    ss = false,
)
    nb = ss ? n_par.nb : n_par.nb - 1
    nk = ss ? n_par.nk : n_par.nk - 1
    nh = ss ? n_par.nh : n_par.nh - 1
    nCop = ss ? n_par.nb * n_par.nk * n_par.nh : nIdxDistr
    return CopulaTwoAssetsIndexes(
        1:nCop, # Copula
        nCop .+ (1:nb), # distr_b
        nCop .+ (nb) .+ (1:nk), # distr_k
        nCop .+ (nb) .+ (nk) .+ (1:nh), # distr_h
    )
end

"""
    create_distribution_indexes(n_par, nIdxDistr::Int, model::CompleteMarkets, distribution_states)

Create empty distribution indexes for CompleteMarkets model (no heterogeneity, no distribution).
"""
function create_distribution_indexes(
    n_par,
    nIdxDistr::Int,
    model::CompleteMarkets,
    distribution_states::Union{CDFStates,CopulaStates};
    ss = false,
)
    # CompleteMarkets requires productivity
    nh = ss ? n_par.nh : nIdxDistr - 1
    return RepAgentIndexes(1:nh)
end

"""
    create_value_function_indexes(model::OneAsset, n_states, nIdxDistr, nIdxValues)

Create value function indexes for OneAsset model.
"""
function create_value_function_indexes(model::OneAsset, n_states, nIdxDistr, nIdxValues)
    return ValueFunctionsOneAssetIndexes(
        (nIdxDistr + n_states + 1):(nIdxDistr + n_states + sum(nIdxValues; init = 0)), # Value function for b
    )
end

"""
    create_value_function_indexes(model::TwoAsset, n_states, nIdxDistr, nIdxValues)

Create value function indexes for TwoAsset model.
"""
function create_value_function_indexes(model::TwoAsset, n_states, nIdxDistr, nIdxValues)
    return ValueFunctionsTwoAssetsIndexes(
        (nIdxDistr + n_states + 1):(nIdxDistr + n_states + nIdxValues[1]), # Value function for b
        (nIdxDistr + n_states + nIdxValues[1] + 1):(nIdxDistr + n_states + sum(
            nIdxValues;
            init = 0,
        )), # Value function for k
    )
end

"""
    create_value_function_indexes(model::OneAsset, n_states, nIdxDistr, nIdxValues)

Create value function indexes for CompleteMarkets model.
"""
function create_value_function_indexes(
    model::CompleteMarkets,
    n_states,
    nIdxDistr,
    nIdxValues,
)
    if sum(nIdxValues; init = 0) > 0
        return ValueFunctionsCompleteMarketsIndexes(
            (nIdxDistr + n_states + 1):(nIdxDistr + n_states + sum(nIdxValues; init = 0)), # Value function for b
        )
    else
        return ValueFunctionsCompleteMarketsIndexes(Int[])
    end
end

"""
    @asset_vars(model_type)

Generate asset variable tuple based on model type, only accessing variables that exist.
"""
macro asset_vars(model_type)
    quote
        if typeof($(esc(model_type))) == OneAsset
            [$(esc(:TotalAssets)), $(esc(:BD))]
        elseif typeof($(esc(model_type))) == TwoAsset
            [$(esc(:TotalAssets)), $(esc(:BD)), $(esc(:K)), $(esc(:B)), $(esc(:q))]
        else
            []
        end
    end
end
