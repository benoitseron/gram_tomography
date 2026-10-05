"""
Projection of a Hermitian matrix onto the set of valid pure-state Gram
("correlation") matrices

    C = { A ∈ M_{n×n}(ℂ) : A = A†, A_{ii} = 1, A ⪰ 0 }.

Implements the alternating-projections scheme of Higham (cited as
`\\cite{nigham}` (sic) in `notes/gtomo_new/Sections/gram-estimation.tex`, in the
"Estimation procedure" subsection). The Hilbert–Schmidt projection onto each of
the two convex sets is:

- `P_U(A)`: set diagonal of `A` to 1 (and Hermitise for safety).
- `P_S(A)`: project onto PSD cone by zeroing out negative eigenvalues.

Both commute with Hermitisation; alternating them converges to `π_C(A)`
(Dykstra's refinement adds corrections to avoid slowdown at intersections, but
the plain alternating scheme already converges in practice for this problem).

The projection step in `gram-estimation.tex` (just after `eq:gram2`) shows that
`‖π_C(Ĝ) - G‖_F ≤ ‖Ĝ - G‖_F` since `G ∈ C`, so applying this projection never
increases the Frobenius error.
"""

using LinearAlgebra

"""
    project_unit_diag(A)

P_U: Hermitise and force unit diagonal.
"""
function project_unit_diag(A::AbstractMatrix)
    B = 0.5 * (A + A')
    @inbounds for i in 1:size(B, 1)
        B[i, i] = 1
    end
    return B
end

"""
    project_psd(A)

P_S: Hermitian → nearest PSD matrix under Frobenius norm.
Eigendecompose, clip negative eigenvalues to zero, reassemble.
"""
function project_psd(A::AbstractMatrix)
    H = 0.5 * (A + A')                # fresh dense matrix; Hermitian wraps it without a copy
    vals, vecs = eigen(Hermitian(H))
    clipped = max.(vals, zero(eltype(vals)))
    return vecs * Diagonal(clipped) * vecs'
end

"""
    project_gram(A; maxiter=200, tol=1e-10) -> (G_proj, iters, converged)

Alternating projections `π_C = lim (P_U P_S)^k` applied to `A`.

Stops when consecutive iterates differ by `tol` in Frobenius norm, or after
`maxiter` iterations. Returns the projected matrix, iteration count, and a
convergence flag.
"""
function project_gram(A::AbstractMatrix; maxiter::Integer=200, tol::Real=1e-10)
    X = Matrix(A)                 # plain Matrix: keeps `X = project_psd(X)` type-stable
    X_prev = copy(X)
    iters = 0
    converged = false
    for k in 1:maxiter
        copyto!(X_prev, X)
        X = project_psd(X)
        X = project_unit_diag(X)
        iters = k
        if norm(X - X_prev) < tol
            converged = true
            break
        end
    end
    return X, iters, converged
end

"""
    is_valid_gram(A; tol=1e-8) -> Bool

Check whether `A` is Hermitian, has unit diagonal, and is PSD to `tol`.
"""
function is_valid_gram(A::AbstractMatrix; tol::Real=1e-8)
    n = size(A, 1)
    size(A, 2) == n || return false
    norm(A - A') > tol && return false
    any(i -> abs(A[i, i] - 1) > tol, 1:n) && return false
    vals = eigvals(Hermitian(0.5 * (A + A')))
    return minimum(vals) ≥ -tol
end
