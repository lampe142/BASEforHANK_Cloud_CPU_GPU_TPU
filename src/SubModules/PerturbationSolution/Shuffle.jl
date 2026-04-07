"""
    shuffleMatrix(distr::AbstractArray)

Construct a transformation matrix `Γ` to reduce the dimensionality of a probability
distribution vector by exploiting the unit sum constraint (probabilities sum to 1).

This matrix maps a reduced distribution vector (dimension N-1) back to the full distribution
vector (dimension N) and helps in computing derivatives with respect to the distribution
while respecting the linear dependency.

The logic is based on: `P_N = 1 - sum(P_1 ... P_{N-1})`

The returned matrix `Γ` has the property that for a perturbation `dx` of the reduced states:
`d(FullDistribution) = Γ * dx`

# Arguments

  - `distr`: The steady-state distribution array (3D, 2D, or 1D). The dimensions correspond
    to (liquid assets `b`, illiquid assets `k`, human capital `h`) or subsets thereof.

# Returns

  - `Γ::Vector{Matrix{Float64}}`: A vector of transformation matrices, one for each
    dimension of the input distribution.

      + For a 3D input (nb, nk, nh), it returns 3 matrices for `b`, `k`, and `h` margins.
      + For a 2D input (nb, nh), it returns 2 matrices for `b` and `h`.
      + For a 1D input (nh), it returns 1 matrix.
"""
function shuffleMatrix(distr::Array{Float64,3})
    nb, nk, nh = size(distr)
    distr_b = sum(sum(distr; dims = 3); dims = 2) ./ sum(distr[:])
    distr_k = sum(sum(distr; dims = 3); dims = 1) ./ sum(distr[:])
    distr_h = sum(sum(distr; dims = 2); dims = 1) ./ sum(distr[:])
    Γ = Array{Array{Float64,2},1}(undef, 3)
    Γ[1] = zeros(Float64, (nb, nb - 1))
    Γ[2] = zeros(Float64, (nk, nk - 1))
    Γ[3] = zeros(Float64, (nh, nh - 1))
    for j = 1:(nb - 1)
        Γ[1][:, j] = -distr_b[:]
        Γ[1][j, j] = 1 - distr_b[j]
        Γ[1][j, j] = Γ[1][j, j] - sum(Γ[1][:, j])
    end
    for j = 1:(nk - 1)
        Γ[2][:, j] = -distr_k[:]
        Γ[2][j, j] = 1 - distr_k[j]
        Γ[2][j, j] = Γ[2][j, j] - sum(Γ[2][:, j])
    end
    for j = 1:(nh - 1)
        Γ[3][:, j] = -distr_h[:]
        Γ[3][j, j] = 1 - distr_h[j]
        Γ[3][j, j] = Γ[3][j, j] - sum(Γ[3][:, j])
    end

    return Γ
end

function shuffleMatrix(distr::Array{Float64,2})
    nb, nh = size(distr)
    distr_b = sum(distr; dims = 2) ./ sum(distr[:])
    distr_h = sum(distr; dims = 1) ./ sum(distr[:])
    Γ = Array{Array{Float64,2},1}(undef, 2)
    Γ[1] = zeros(Float64, (nb, nb - 1))
    Γ[2] = zeros(Float64, (nh, nh - 1))
    for j = 1:(nb - 1)
        Γ[1][:, j] = -distr_b[:]
        Γ[1][j, j] = 1 - distr_b[j]
        Γ[1][j, j] = Γ[1][j, j] - sum(Γ[1][:, j])
    end
    for j = 1:(nh - 1)
        Γ[2][:, j] = -distr_h[:]
        Γ[2][j, j] = 1 - distr_h[j]
        Γ[2][j, j] = Γ[2][j, j] - sum(Γ[2][:, j])
    end

    return Γ
end

function shuffleMatrix(distr::Array{Float64,1})
    nh = length(distr)
    Γ = zeros(Float64, (nh, nh - 1))
    for j = 1:(nh - 1)
        Γ[:, j] = -distr[:]
        Γ[j, j] = 1.0 - distr[j]
        Γ[j, j] = Γ[j, j] - sum(Γ[:, j])
    end
    return [Γ]
end
