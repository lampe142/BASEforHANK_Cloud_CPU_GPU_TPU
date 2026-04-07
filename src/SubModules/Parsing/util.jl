"""
    find_field_with_value(struct_instance, field_value)

Find the names of fields in a struct instance that contain a specific value.

Parameters:

  - `struct_instance`: An instance of a Julia struct.
  - `field_value`: The value to search for within the fields of the struct.

Returns:

  - A list of field names that contain the specified value, or an empty list if no such
    field is found.
"""
function find_field_with_value(struct_instance, field_value)
    matching_fields = Symbol[]
    for field_name in fieldnames(typeof(struct_instance))
        field_val = getfield(struct_instance, field_name)
        if isa(field_val, Int64)
            if field_value == field_val || field_value in field_val
                push!(matching_fields, field_name)
            end
        else
            SS_suffix = occursin("SS", string(field_name)) ? "SS" : ""
            for field_name2 in fieldnames(typeof(field_val))
                field_val2 = getfield(field_val, field_name2)
                if field_value in field_val2
                    push!(matching_fields, Symbol(string(field_name2) * SS_suffix))
                end
            end
        end
    end

    if isempty(matching_fields)
        @warn "No field with value $field_value found!"
    end

    return matching_fields
end

"""
    find_field_with_value(struct_instance, field_value, ss)

Wrapper function for `find_field_with_value` that allows for selection based on the
presence of "SS" in the field name.
"""
function find_field_with_value(struct_instance, field_value, ss)
    matching_fields = find_field_with_value(struct_instance, field_value)
    if length(matching_fields) == 0
        @warn "No field with value $field_value found!"
    elseif length(matching_fields) == 1
        return matching_fields[1]
    elseif length(matching_fields) == 2
        if ss == true
            for field in matching_fields
                if occursin("SS", string(field))
                    return field
                end
            end
        else
            for field in matching_fields
                if !occursin("SS", string(field))
                    return field
                end
            end
        end
    elseif length(matching_fields) > 2
        @error "More than two fields with value $field_value found!"
    end
end

"""
    mapround(b; digits = 4)

The `mapround` function rounds every element of a matrix `b` to `digits` number of decimal
places.

# Arguments

  - `b`: a matrix of numbers to be rounded
  - `digits`: number of decimal places to round to (default is 4)

# Returns

A matrix with the same shape as `b` where every element has been rounded to `digits` number
of decimal places.
"""
function mapround(b; digits = 4)
    map(x -> round.(x; digits = digits), b)
end

"""
    valFunc_from_vec(valueFunction_vec, model)

Convert a flattened value-function vector into the appropriate ValueFunction struct
depending on the concrete `model` type.

Returns

  - `ValueFunctionsOneAsset` when `model::OneAsset`
  - `ValueFunctionsTwoAsset` when `model::TwoAsset`
"""
valFunc_from_vec(valueFunction_vec, model::OneAsset) =
    vec_to_struct(ValueFunctionsOneAsset, valueFunction_vec)
valFunc_from_vec(valueFunction_vec, model::TwoAsset) =
    vec_to_struct(ValueFunctionsTwoAssets, valueFunction_vec)

"""
Construct a `PolicyFunctions` object with uninitialized arrays matching the shape and type of `template`.
"""
PolicyFunctionsTwoAssets{T}(template::T) where {T} = PolicyFunctionsTwoAssets(
    similar(template),  # x_a
    similar(template),  # b_a
    similar(template),  # k_a
    similar(template),  # x_n
    similar(template),  # b_n
    similar(template),  # x_tilde_n
    similar(template),  # b_tilde_n
)
PolicyFunctionsOneAsset{T}(template::T) where {T} = PolicyFunctionsOneAsset(
    similar(template),  # x_n
    similar(template),  # b_n
    similar(template),  # x_tilde_n
    similar(template),  # b_tilde_n
)
PolicyFunctionsCompleteMarkets{T}(template::T) where {T} =
    PolicyFunctionsCompleteMarkets{T}()
"""
Construct a `ValueFunctions` object with uninitialized arrays matching the shape and type of `template`.
"""
ValueFunctionsTwoAssets{T}(template::T) where {T} = ValueFunctionsTwoAssets(
    similar(template),  # b
    similar(template),  # k
)
ValueFunctionsOneAsset{T}(template::T) where {T} = ValueFunctionsOneAsset(
    similar(template),  # b
)
ValueFunctionsCompleteMarkets{T}(template::T) where {T} = ValueFunctionsCompleteMarkets()

"""
    similar(vf::ValueFunctions)

Create a new `ValueFunctions` object with similar but uninitialized arrays.
"""
function Base.similar(vf::ValueFunctionsTwoAssets{T}) where {T}
    return ValueFunctionsTwoAssets(
        similar(vf.b),  # liquid asset value function
        similar(vf.k),  # illiquid asset value function
    )
end

function Base.similar(vf::ValueFunctionsOneAsset{T}) where {T}
    return ValueFunctionsOneAsset(
        similar(vf.b),  # liquid asset value function
    )
end

function Base.similar(vf::ValueFunctionsCompleteMarkets{T}) where {T}
    return ValueFunctionsCompleteMarkets(similar(vf.b))
end

"""
    copy(vf::ValueFunctions)

Create a deep copy of a `ValueFunctions` object with all arrays copied.
"""
function Base.copy(vf::ValueFunctionsTwoAssets{T}) where {T}
    return ValueFunctionsTwoAssets(
        copy(vf.b),  # liquid asset value function
        copy(vf.k),  # illiquid asset value function
    )
end

function Base.copy(vf::ValueFunctionsOneAsset{T}) where {T}
    return ValueFunctionsOneAsset(
        copy(vf.b),  # liquid asset value function
    )
end

function Base.copy(vf::ValueFunctionsCompleteMarkets{T}) where {T}
    return ValueFunctionsCompleteMarkets(copy(vf.b))
end

"""
    similar(pf::PolicyFunctions)

Create a new `PolicyFunctions` object with similar but uninitialized arrays.
"""
function Base.similar(pf::PolicyFunctionsTwoAssets{T}) where {T}
    return PolicyFunctionsTwoAssets(
        similar(pf.x_a_star),  # composite consumption, adjustment case
        similar(pf.b_a_star),  # liquid asset policy, adjustment case
        similar(pf.k_a_star),  # illiquid asset policy, adjustment case
        similar(pf.x_n_star),  # composite consumption, non-adjustment case
        similar(pf.b_n_star),  # liquid asset policy, non-adjustment case
        similar(pf.x_tilde_n), # endogenous grid consumption, non-adjustment case
        similar(pf.b_tilde_n), # endogenous grid liquid asset, non-adjustment case
    )
end

function Base.similar(pf::PolicyFunctionsOneAsset{T}) where {T}
    return PolicyFunctionsOneAsset(
        similar(pf.x_n_star),  # composite consumption
        similar(pf.b_n_star),  # liquid asset policy
        similar(pf.x_tilde_n), # endogenous grid consumption
        similar(pf.b_tilde_n), # endogenous grid liquid asset
    )
end

function Base.similar(pf::PolicyFunctionsCompleteMarkets{T}) where {T}
    return PolicyFunctionsCompleteMarkets{T}()
end

# =============================================================================
# Similar functions for DistributionValues types
# =============================================================================

"""
    Base.similar(cdf::CDF)

Create a similar CDF structure with uninitialized arrays of the same size and type.
"""
function Base.similar(cdf::CDF)
    return CDF(similar(cdf.CDF))
end

"""
    Base.similar(cop::CopulaOneAsset)

Create a similar CopulaOneAsset structure with uninitialized arrays of the same size and type.
"""
function Base.similar(cop::CopulaOneAsset)
    return CopulaOneAsset(similar(cop.COP), similar(cop.b), similar(cop.h))
end

"""
    Base.similar(cop::CopulaTwoAssets)

Create a similar CopulaTwoAssets structure with uninitialized arrays of the same size and type.
"""
function Base.similar(cop::CopulaTwoAssets)
    return CopulaTwoAssets(similar(cop.COP), similar(cop.b), similar(cop.k), similar(cop.h))
end

# Set distribution structures based on input, model and marginals type
set_distribution(COP::AbstractArray, ::TwoAsset, ::CopulaStates, ::LinearTransition) =
    CopulaPDFsTwoAssets(
        COP,
        cdf_to_pdf(COP[:, end, end]),
        cdf_to_pdf(COP[end, :, end]),
        cdf_to_pdf(COP[end, end, :]),
    )
set_distribution(COP::AbstractArray, ::OneAsset, ::CopulaStates, ::LinearTransition) =
    CopulaPDFsOneAsset(COP, cdf_to_pdf(COP[:, end]), cdf_to_pdf(COP[end, :]))
set_distribution(COP::AbstractArray, ::OneAsset, ::CopulaStates, ::NonLinearTransition) =
    CopulaCDFsOneAsset(COP, COP[:, end], COP[end, :])
set_distribution(
    cdf::AbstractArray,
    ::Union{OneAsset,TwoAsset},
    ::CDFStates,
    ::TransitionType,
) = CDF(cdf)
set_distribution(
    cdf::AbstractArray,
    ::CompleteMarkets,
    ::DistributionStateType,
    ::TransitionType,
) = RepAgent(cdf)
# Set Copula structures based on input, model and marginals type (type-based dispatch)
set_copula(COP, b, k, h, ::LinearTransition) = CopulaPDFsTwoAssets(COP, b, k, h)
set_copula(COP, b, k, h, ::NonLinearTransition) = CopulaCDFsTwoAssets(COP, b, k, h)
set_copula(COP, b, h, ::LinearTransition) = CopulaPDFsOneAsset(COP, b, h)
set_copula(COP, b, h, ::NonLinearTransition) = CopulaCDFsOneAsset(COP, b, h)

# Get CDF or PDF from marginals based on marginals type
get_CDF(
    marginal::AbstractVector,
    ::Type{T},
) where {T<:Union{CopulaPDFsOneAsset,CopulaPDFsTwoAssets}} = pdf_to_cdf(marginal)
get_CDF(
    marginal::AbstractVector,
    ::Type{T},
) where {T<:Union{CopulaCDFsOneAsset,CopulaCDFsTwoAssets}} = marginal

get_PDF(
    marginal::AbstractVector,
    ::Type{T},
) where {T<:Union{CopulaPDFsOneAsset,CopulaPDFsTwoAssets}} = marginal
get_PDF(
    marginal::AbstractVector,
    ::Type{T},
) where {T<:Union{CopulaCDFsOneAsset,CopulaCDFsTwoAssets}} = cdf_to_pdf(marginal)

# Get last index along all dimensions except the last one (equivalent to a[end, end, :] or a[end, :])
last_slice_except_last_dim(a) =
    a[ntuple(i -> i == ndims(a) ? Colon() : lastindex(a, i), ndims(a))...]

get_PDF_h(distr::Union{CopulaPDFsOneAsset,CopulaPDFsTwoAssets}) = distr.h
get_PDF_h(distr::Union{CopulaCDFsOneAsset,CopulaCDFsTwoAssets}) = cdf_to_pdf(distr.h)
get_PDF_h(distr::CDF) = cdf_to_pdf(last_slice_except_last_dim(distr.CDF))
get_PDF_h(distr::RepAgent) = cdf_to_pdf(distr.h)
get_slice_h(a::AbstractArray, entrepreneur::Val{true}) = a[1:(end - 1)]
get_slice_h(a::AbstractArray, entrepreneur::Val{false}) = a

get_joint_CDF(distr::CDF) = distr.CDF
get_joint_CDF(distr::Union{CopulaOneAsset,CopulaTwoAssets}) = distr.COP

get_Dindex(d_indexes::CDFIndexes) = d_indexes.CDF
get_Dindex(d_indexes::Union{CopulaOneAssetIndexes,CopulaTwoAssetsIndexes}) = d_indexes.COP
