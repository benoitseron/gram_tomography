"""
Reconstruction of state vectors from a Gram matrix, up to the unitary freedom
R ∈ U(d) as described in the manuscript's "Reconstructing state-vectors"
discussion (`notes/Gram_Tomography.pdf`; Powers–Størmer bound in the
CLAUDE.md Key Equations table).

Given a valid Gram matrix Ĝ = B† B of rank ≤ d, we want B ∈ ℂ^{d × n} with
columns `ψ_i` so that Ĝ_{ij} = ⟨ψ_i | ψ_j⟩. Any two such factorisations differ
by a left-multiplication by a d × d unitary, which is the equivalence class
the notes mention.

Conjugate / adjoint conventions (see `CLAUDE.md`):

- `B' == adjoint(B) == conj(transpose(B))`. We never write `transpose(B)` on a
  complex matrix when a dagger is intended. All Hermitisations use
  `0.5 * (A + A')`.
- With `B ∈ ℂ^{d × n}` and columns `ψ_i`, `G = B' * B` (NOT `B * B'`, which is
  the `d × d` density-like operator `Σ_i ψ_i ψ_i†`).
- Orthogonal-Procrustes derivation (tying the two equivalent forms):
  minimise `f(R) = ‖R B₁ − B₂‖_F²`. Expanding,
      f = const − 2 Re tr(B₂' * R * B₁) = const − 2 Re tr(R * B₁ * B₂').
  Define `A := B₁ * B₂'` (i.e. `B₁ B₂†` in math). SVD `A = U Σ V'`. Then the
  optimum is `R_opt = V * U'` (note the dagger on `U`, not on `V`). In code we
  write it with `M = B₂ * B₁'` as the SVD input because `svd(A')` returns
  `(V, Σ, U')` — see inline comment in `align_unitary`.

We provide two reconstructors:

- `reconstruct_eigen(Ĝ, d)`  — via spectral decomposition Ĝ = U D U†, returning
  B = √D · U† truncated to the top d eigenmodes. This is the method used to
  derive the Powers–Størmer bound.
- `reconstruct_cholesky(Ĝ)` — Cholesky factor (only when Ĝ is full rank in
  dimension n; produces an n × n triangular factor, useful for verification).

We also expose `powers_stormer_bound(A, B)` which returns the analytical
bound `n^{1/4} · ‖A - B‖_2^{1/2}` from L225.
"""

using LinearAlgebra

"""
    reconstruct_eigen(Ĝ::AbstractMatrix, d::Integer=size(Ĝ,1))

Return `B ∈ ℂ^{d × n}` with `Ĝ ≈ B† B`.

If `d < n`, truncates to the `d` largest eigenvalues (rank-`d` approximation).
Negative eigenvalues are clipped to zero (caller should pre-project Ĝ via
`project_gram` to be safe).
"""
function reconstruct_eigen(Ĝ::AbstractMatrix, d::Integer=size(Ĝ, 1))
    n = size(Ĝ, 1)
    @assert size(Ĝ, 2) == n
    H = 0.5 * (Ĝ + Ĝ')                # fresh dense matrix; Hermitian wraps it without a copy
    vals, vecs = eigen(Hermitian(H))
    perm = partialsortperm(vals, 1:min(d, n), rev=true)   # d largest, no full sort
    λ = max.(vals[perm], zero(eltype(vals)))
    V = vecs[:, perm]
    # B is d × n; columns are the vectors ψ_i.
    # Ĝ = V Λ V† = (√Λ V†)† (√Λ V†), so B = √Λ V†.
    B = Diagonal(sqrt.(λ)) * V'
    return B
end

"""
    reconstruct_cholesky(Ĝ::AbstractMatrix)

Return upper triangular `B` with `Ĝ = B† B`. Requires `Ĝ` positive definite.
"""
reconstruct_cholesky(Ĝ::AbstractMatrix) = cholesky(Hermitian(0.5 * (Ĝ + Ĝ'))).U

"""
    powers_stormer_bound(A::AbstractMatrix, B::AbstractMatrix)

Analytical Powers–Størmer bound (CLAUDE.md Key Equations table):

    ‖√A - √B‖_2 ≤ n^{1/4} · ‖A - B‖_2^{1/2}

assuming A, B ⪰ 0. Returns the right-hand side.
"""
function powers_stormer_bound(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A) == size(B)
    n = size(A, 1)
    R = real(float(promote_type(eltype(A), eltype(B))))   # follow input precision
    return R(n)^(one(R) / 4) * sqrt(opnorm(A - B, 2))
end

"""
    align_unitary(B1::AbstractMatrix, B2::AbstractMatrix)

Given two factorisations `B1, B2 ∈ ℂ^{d × n}` of Gram matrices that should be
compared up to the left-unitary freedom `B ↦ R B` (the residual `R ∈ U(d)`), return
`(R, aligned_B1)` where `R ∈ U(d)` minimises `‖R B1 - B2‖_F` via the orthogonal
Procrustes solution (SVD of B2 B1†).
"""
function align_unitary(B1::AbstractMatrix, B2::AbstractMatrix)
    @assert size(B1) == size(B2)
    # Orthogonal Procrustes: R_opt = V * U' where B1*B2' = U Σ V'.
    # Equivalently B2*B1' = V Σ U'; with SVD(M).U = V and SVD(M).Vt = U',
    # this gives R_opt = SVD(M).U * SVD(M).Vt. Never use `transpose` here:
    # `B1'` is the adjoint (conjugate transpose).
    M = B2 * B1'
    F = svd(M)
    R = F.U * F.Vt
    return R, R * B1
end
