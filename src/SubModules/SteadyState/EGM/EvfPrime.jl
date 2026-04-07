"""
    expected_marginal_values(Π, vfPrime, n_par, beta_factor)

Compute expected marginal value functions `EvfPrime` given transition matrix `Π`
(idiosyncratic risk) and `beta_factor` (discount factor shock).
"""
function expected_marginal_values(
    Π,
    vfPrime::ValueFunctionsCompleteMarkets,
    n_par::NumericalParameters,
    beta_factor = 1.0,
)
    template = similar(zeros(eltype(Π), size(vfPrime.b)))
    EvfPrime = ValueFunctionsCompleteMarkets(template)
    return EvfPrime
end

function expected_marginal_values(
    Π,
    vfPrime::ValueFunctionsOneAsset,
    n_par::NumericalParameters,
    beta_factor = 1.0,
)
    template = similar(zeros(eltype(Π), size(vfPrime.b)))
    EvfPrime = ValueFunctionsOneAsset(template)
    EvfPrime.b = vfPrime.b .+ zeros(eltype(Π), 1)[1]
    EvfPrime.b = vfPrime.b * Π' .* beta_factor
    return EvfPrime
end

function expected_marginal_values(
    Π,
    vfPrime::ValueFunctionsTwoAssets,
    n_par::NumericalParameters,
    beta_factor = 1.0,
)
    template = similar(zeros(eltype(Π), size(vfPrime.b)))
    EvfPrime = ValueFunctionsTwoAssets(template, template)

    EvfPrime.b = vfPrime.b .+ zeros(eltype(Π), 1)[1]
    EvfPrime.k = vfPrime.k .+ zeros(eltype(Π), 1)[1]
    @views @inbounds begin
        for bb = 1:(n_par.nb)
            EvfPrime.k[bb, :, :] .= (EvfPrime.k[bb, :, :] * Π') .* beta_factor
            EvfPrime.b[bb, :, :] .= (EvfPrime.b[bb, :, :] * Π') .* beta_factor
        end
    end

    return EvfPrime
end
