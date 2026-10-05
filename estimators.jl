"""
Sample-based estimators for Bargmann invariants and derived quantities.

Implements the estimators analysed in `notes/Gram_Tomography.pdf`,
Appendix A (`Sections/analysis-of-estimators.tex`; bias/variance of √B̂,
phase) and the Thm 1 bound (eq. 40–42):

    B̂_π       — unbiased estimator of B_π from N samples, variance σ²/N.
    √̂B_{ij}   := √(B̂_{ij})                                              (Eq. 32, 34)
    ê^{iφ}    := B̂_π / |B̂_π|                                            (Eq. 33)

The bias/variance formulas (Prop. 2/3, Lemma 1 of the manuscript) are predicted
by Taylor expansion and verified numerically in `tests.jl`.

For the purposes of numerical experiments we model `B̂_π` as
    B̂_π = B_π + (σ/√N) · Z,          Z ∼ CN(0, 1) (complex standard normal).
This reproduces the σ²/N variance assumption stated by the manuscript. In
this synthetic-noise model, estimators on different paths are *independent*
by construction, so the covariance term `K(δ)` (eq. 42) can be dropped in
the Thm 1 bound — see `frobenius_mse_bound_thm1` below.
"""

using Random
using Statistics

"""
    simulate_bargmann_estimate(B_true::Number, σ::Real, N::Integer;
                               rng=Random.default_rng(), complex_noise::Bool=true)

Return one realisation `B̂ = B_true + (σ/√N) · Z` with `Z ∼ CN(0,1)`
(so `Var(B̂) = σ²/N`). Set `complex_noise=false` to use real Gaussian noise,
appropriate when the estimator is known to be real (e.g. |G_{ij}|² for
tree edges).
"""
function simulate_bargmann_estimate(B_true::Number, σ::Real, N::Integer;
                                    rng=Random.default_rng(), complex_noise::Bool=true)
    scale = σ / sqrt(N)
    if complex_noise
        z = (randn(rng) + im*randn(rng)) / sqrt(2)
        return B_true + scale * z
    else
        return B_true + scale * randn(rng)
    end
end

"""
    bias_sqrt_B(B::Real, σ::Real, N::Integer)

Leading-order bias of `√B̂` predicted by Taylor expansion (Eq. (A3) of
`Gram_Tomography.pdf`; `eq:bias1` in `Sections/analysis-of-estimators.tex`):

    E[√B̂] - √B ≈ -σ² / (8 N · B^{3/2})
"""
bias_sqrt_B(B::Real, σ::Real, N::Integer) = -σ^2 / (8 * N * B^(3/2))

"""
    var_sqrt_B(B::Real, σ::Real, N::Integer)

Leading-order variance of `√B̂` (Eq. (A4) of `Gram_Tomography.pdf`):

    Var(√B̂) ≈ σ² / (4 N · B)
"""
var_sqrt_B(B::Real, σ::Real, N::Integer) = σ^2 / (4 * N * B)

"""
    var_phase(Bπ_magnitude::Real, σ::Real, N::Integer)

Leading-order variance of `ê^{iφ} = B̂_π / |B̂_π|` (phase estimator
`eq:estimator-phase`; Appendix-A Taylor analysis, cf. Eq. (A11)):

    Var(ê^{iφ}) ≈ σ² / (2 N · |B_π|²)
"""
var_phase(Bπ_magnitude::Real, σ::Real, N::Integer) = σ^2 / (2 * N * Bπ_magnitude^2)

# --- Thm 1 bound (Gram_Tomography.pdf, eq. 40–42) ---------------------------

"""
    V_delta_thm1(δ::Real, M::Real, m::Integer, σ::Real, N::Integer;
                 ε::Real=0.0, include_covariance::Bool=false) -> Real

Per-entry variance proxy `V(δ)` from eq. 41 (literal transcription, with `δ`
the user-chosen removal threshold — `δ_min` defined in eq. 37 does NOT appear
in the bound formulas themselves, only in the derivation eq. 39):

    V(δ)  ≤  [ 4/√δ + (M/√δ)·4m ] · ε
            + [ 1/(2 δ^{3/2}) + 1/(4 δ) + (1/2)(M/√δ)² ] · σ²/N
            + 2·K(δ)

where `M = M_T_thm1(T, W)` (sqrt-inside-product, eq. 38), `m = m_T(T)`
(hop-count diameter + 1), and `K(δ)` is the Cauchy–Schwarz covariance term
(eq. 42) — droppable when estimators on different paths are independent,
which is the case for our synthetic noise model.

Pure-state regime corresponds to `ε = 0` (ε-pieces drop out).

The bound is monotone-decreasing in `δ` on `(0, δ_min]` and saturates at
`δ = δ_min` (where the truncation term `n(δ)·δ` of eq. 40 first hits zero).
"""
function V_delta_thm1(δ::Real, M::Real, m::Integer, σ::Real, N::Integer;
                     ε::Real=0.0, include_covariance::Bool=false)
    sqd  = sqrt(δ)
    MoS  = M / sqd
    eps_piece   = (4 / sqd + MoS * 4 * m) * ε
    noise_piece = (1/(2 * δ^(3/2)) + 1/(4 * δ) + 0.5 * MoS^2) * (σ^2 / N)
    V = eps_piece + noise_piece
    if include_covariance
        K = K_delta_thm1(δ, M, m, σ, N; ε=ε)
        V += 2 * K
    end
    return V
end

"""
    K_delta_thm1(δ::Real, M::Real, m::Integer, σ::Real, N::Integer;
                 ε::Real=0.0) -> Real

Cauchy–Schwarz covariance term from eq. 42 (literal transcription):

    K(δ) = √( [ (M/√δ)·4m·ε + (1/2)(M/√δ)²·σ²/N ] · [ (1/(4 δ))·σ²/N ] )

Worst-case bound on `Cov(X̂, Ŷ)` between the magnitude-√B̂ estimator and the
phase estimator on the same edge. Vanishes when the two are independent
(true for our synthetic noise model, which draws fresh CN(0,1) per path).
"""
function K_delta_thm1(δ::Real, M::Real, m::Integer, σ::Real, N::Integer;
                      ε::Real=0.0)
    sqd = sqrt(δ)
    MoS = M / sqd
    inner_a = MoS * 4 * m * ε + 0.5 * MoS^2 * (σ^2 / N)
    inner_b = (1 / (4 * δ)) * (σ^2 / N)
    return sqrt(inner_a * inner_b)
end

"""
    frobenius_mse_bound_thm1(n::Int, n_δ::Integer, δ::Real,
                             M::Real, m::Integer, σ::Real, N::Integer;
                             ε::Real=0.0, include_covariance::Bool=false) -> Real

Thm 1 Frobenius MSE bound (eq. 40):

    E‖G - Ĝ‖²₂  ≤  n(δ)·δ  +  n² · V(δ)

with `V(δ)` from `V_delta_thm1`. Parameters:

- `n_δ = n(δ)` is the number of edges removed (those with `|B̂_ij| < δ`),
- `δ` is the user-chosen removal threshold,
- `M = M_T_thm1(T, W)` (eq. 38), `m = m_T(T)` (eq. 41).

Pure-state regime: pass `ε = 0` (default). Drop `K(δ)` (default) when noise
on different paths is independent.

Optimal `δ`: the manuscript advises taking `δ = δ_min` (smallest surviving
Bargmann invariant) to set `n(δ) = 0`, in which case the bound collapses to
`n² · V(δ_min)`. Strictly, the optimum may be larger when removing weak edges
buys back more in `V(δ)` than it costs in `n(δ)·δ`.
"""
function frobenius_mse_bound_thm1(n::Int, n_δ::Integer, δ::Real,
                                  M::Real, m::Integer, σ::Real, N::Integer;
                                  ε::Real=0.0, include_covariance::Bool=false)
    truncation = n_δ * δ
    V = V_delta_thm1(δ, M, m, σ, N;
                     ε=ε, include_covariance=include_covariance)
    return truncation + n^2 * V
end
