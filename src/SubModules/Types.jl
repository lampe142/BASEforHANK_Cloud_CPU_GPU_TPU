module Types

abstract type AbstractMacroModel end
struct OneAsset <: AbstractMacroModel end
struct TwoAsset <: AbstractMacroModel end
struct CompleteMarkets <: AbstractMacroModel end

abstract type PolicyFunctions{T<:AbstractArray} end
abstract type ValueFunctions{T<:AbstractArray} end
abstract type TransitionMatrices{T<:AbstractArray} end

abstract type TransitionType end
struct NonLinearTransition <: TransitionType end
struct LinearTransition <: TransitionType end

abstract type DistributionValues end
abstract type CopulaOneAsset <: DistributionValues end
abstract type CopulaTwoAssets <: DistributionValues end

abstract type Indexes end
abstract type DistributionIndexes <: Indexes end
abstract type ValueFunctionIndexes <: Indexes end

abstract type DistributionStateType end
struct CDFStates <: DistributionStateType end
struct CopulaStates <: DistributionStateType end

abstract type Transformations end
struct ParetoTransformation <: Transformations end
struct LinearTransformation <: Transformations end

export AbstractMacroModel,
    TwoAsset,
    OneAsset,
    CompleteMarkets,
    PolicyFunctions,
    ValueFunctions,
    TransitionMatrices,
    TransitionType,
    NonLinearTransition,
    LinearTransition,
    CDFStates,
    CopulaStates,
    DistributionValues,
    Indexes,
    DistributionIndexes,
    ValueFunctionIndexes,
    Transformations,
    ParetoTransformation,
    LinearTransformation,
    CopulaOneAsset,
    CopulaTwoAssets,
    DistributionStateType

end
