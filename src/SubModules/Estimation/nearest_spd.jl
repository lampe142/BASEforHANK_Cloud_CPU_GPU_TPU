
"""
    nearest_spd(A)

Return the nearest symmetric positive definite (SPD) matrix to `A` in Frobenius norm.

Algorithm (Higham):

  - Symmetrize `A` to `B = (A + A')/2`.
  - Compute the symmetric polar factor `H` of `B` via SVD, and set `Ahat = (B + H)/2`.
  - Enforce exact symmetry: `Ahat = (Ahat + Ahat')/2`.
  - If `Ahat` is not numerically PD, add a small multiple of the identity until a Cholesky
    factorization succeeds (capped at 100 tweaks).

# Arguments

  - `A::AbstractMatrix{<:Real}`: square matrix

# Returns

  - `Ahat::Matrix{<:Real}`: symmetric positive definite matrix near `A` in Frobenius norm

Notes:

  - If `A` is already SPD, the output is (up to roundoff) `A`.

  - The SVD-based construction yields the nearest symmetric positive semidefinite matrix;
    the final identity shift ensures positive definiteness, which is often required for
    covariance matrices.
  - See N. J. Higham (1988), “Computing a nearest symmetric positive semidefinite matrix”.

    # symmetrize A into B
"""
function nearest_spd(A)

    # symmetrize A into B
    B = 0.5 .* (A .+ A')
    FU, FS, FVt = LinearAlgebra.LAPACK.gesvd!('N', 'S', copy(B))
    H = FVt' * Diagonal(FS) * FVt

    # get Ahat in the above formula
    Ahat = 0.5 .* (B .+ H)

    # ensure symmetry
    Ahat .= 0.5 .* (Ahat .+ Ahat')

    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    p = false
    k = 0
    count = 1
    while p == false && count < 100
        R = cholesky(Ahat; check = false)
        k += 1
        count = count + 1
        if ~issuccess(R)
            # Ahat failed the chol test. It must have been just a hair off, due to floating
            # point trash, so it is simplest now just to tweak by adding a tiny multiple of
            # an identity matrix.
            mineig = eigmin(Ahat)
            Ahat .+= (-mineig .* k .^ 2 .+ eps(mineig)) .* Diagonal(ones(size(Ahat, 1)))
        else
            p = true
        end
    end

    return Ahat
end
