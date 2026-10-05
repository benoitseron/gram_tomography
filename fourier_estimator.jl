"""
Fourier multivariate-trace estimator of Bargmann invariants (Novo et al. 2026).

Implements the estimator of *"A native linear-optical protocol for estimating
multivariate traces"*, Novo et al. 2026, arXiv:2601.14204 (cached at
`literature/2601.14204_Novo_native_linear_optical_multivariate_trace.tex`),
which is the "Measuring the Bargmann invariant B_π" box of
`notes/gtomo_new/Sections/Accessing_relation_information.tex`:

To measure `B_π = Tr(ρ_{u₁} ρ_{u₂} ⋯ ρ_{u_M})` for an M-cycle of single-photon
internal states, exploit the Fourier diagonalization of the cyclic-shift
interferometer `Ĉ = F̂ D̂ F̂†` (Novo eq. "eq: Fourier evolution"):

1. send one photon per mode, photon `l` carrying internal state `ψ_{u_l}`
   (Gram submatrix `G_cycle = G[π, π]` along the cycle);
2. evolve through the **inverse** Fourier interferometer `F̂†`
   (Novo eq. "eq:output_Fourier": `Ω_out = F̂† Ω F̂`) and photon-count;
3. classically post-process each outcome `S` through
   `f(S) = Σ_{l=0}^{M-1} l·S_l (mod M)` (Novo eq. "eq:f_of_S"); then
   (Novo Theorem 1, `X_k = E[ω^{k·f(S)}]` with `ω = exp(2πi/M)`)

       X₁ = E[ exp((2πi/M)·f(S)) ] = Δ_{1…M} = Tr(ρ₁⋯ρ_M) = B_π,

   so the empirical mean of `exp((2πi/M)·f(S))` is an unbiased estimator
   of the full complex `B_π`. The `mod M` is automatic (M-periodicity of
   the exponential). At `M = 2` (`ω = −1`) this collapses to the HOM
   estimator `B̂ = P(2,0) + P(0,2) − P(1,1) = |⟨ψ₁|ψ₂⟩|²`.

The method handles ANY invariant order M — unlike the correlator (one-shot)
method of `one_shot_estimator.jl`, which supplies orders ≤ 3 only.

The explicit `M = 2` special case — the per-pair HOM coincidence estimator
of `|G_{ij}|²` (50:50 beamsplitter, `B̂_{ij} = 1 − 2 N_coinc/N`) and its
`hom_tomography_magnitudes` driver — lives at the bottom of this file: it is
both the simplest physical magnitude estimator and the comparison baseline
for the general Fourier estimator (`fourier_interferometer(2) == HOM_U_PHYSICS`).

## Sign convention (locked by test — do NOT flip)

`fourier_interferometer(M)` is `F̂†`, i.e. `U[a,b] = exp(−2πi(a−1)(b−1)/M)/√M`
(1-based; minus sign), and the estimator exponent is `+(2πi/M)·Σ_l (l−1)·S_l`.
With this pair, the population estimator equals `bargmann_invariant(G, cycle)`
EXACTLY — i.e. `B̂ → B_π = G_{u₁u₂}⋯G_{u_M u₁}` (no conj), the project
convention of `CLAUDE.md` § "Bargmann invariants — the conjugation trap".
Using `F̂` instead (plus sign in U) yields `conj(B_π)` for M ≥ 3 — verified
numerically; the M = 2 (HOM) case is real and cannot discriminate. The
testset "Fourier: population estimator equals B_π (convention lock)" pins
this down for M = 2…5 with random complex Gram matrices.

As everywhere, the physics-convention unitary crosses into BosonSampling.jl
through `to_bsj(U) = transpose(U)` only (`CLAUDE.md` § "Interferometer
unitary — U vs Uᵀ").
"""

using LinearAlgebra
using Random
using BosonSampling

"""
    fourier_interferometer(M::Integer) -> Matrix{ComplexF64}

The interferometer of the Novo 2026 protocol in the physics convention
(`b†_out = U · a†_in`): the **inverse** Fourier matrix `F̂†`,

    U[a, b] = (1/√M) · exp(−2πi·(a−1)(b−1)/M)        (1-based indices).

This is the adjoint of Novo's `F̂` ("eq: Fourier evolution",
`F a†_j F† = (1/√M) Σ_k ω^{jk} a†_k`, `ω = exp(2πi/M)`); the protocol
evolves the input through `F̂†` ("eq:output_Fourier"). The minus sign is
the choice for which the estimator converges to `B_π` itself, not
`conj(B_π)` — see the module docstring. Symmetric (`Uᵀ = U`), so `to_bsj`
is a no-op for it; `fourier_interferometer(2) == HOM_U_PHYSICS`.
"""
function fourier_interferometer(M::Integer)
    @assert M ≥ 2 "need at least a 2-cycle"
    return BSJ_COMPLEX[exp(-2π * im * (a - 1) * (b - 1) / M) / sqrt(M)
                       for a in 1:M, b in 1:M]
end

"""
    fourier_phase_factor(S::AbstractVector{<:Integer}, M::Integer) -> ComplexF64

The classical post-processing of one photon-count outcome `S` (length-M
occupation vector): `exp((2πi/M)·f(S))` with `f(S) = Σ_l (l−1)·S_l`
(Novo "eq:f_of_S"; the `mod M` is automatic by periodicity). Its
expectation over outcomes is `X₁ = B_π` (Novo Theorem 1).
"""
function fourier_phase_factor(S::AbstractVector{<:Integer}, M::Integer)
    f = sum((l - 1) * S[l] for l in 1:M)
    return exp(2π * im * f / M)
end

"""
    fourier_bargmann_population(G_cycle::AbstractMatrix{<:Complex}) -> ComplexF64

EXACT (population-limit, no sampling noise) value of the Fourier estimator:

    Σ_S p(S) · exp((2πi/M)·f(S))  =  X₁  =  B_π = Tr(ρ_{u₁}⋯ρ_{u_M}),

with `p(S)` the exact M-photon output distribution of the cycle's Gram
submatrix `G_cycle = G[π, π]` through `fourier_interferometer(M)`
(`full_distribution`, BosonSampling.jl convention via `to_bsj`). Equals
`bargmann_invariant(G_cycle, 1:M)` to machine precision — the convention
lock; used for noiseless validation of `fourier_bargmann_estimate`.
"""
function fourier_bargmann_population(G_cycle::AbstractMatrix{<:Complex})
    M = size(G_cycle, 1)
    @assert size(G_cycle, 2) == M "G_cycle must be square"
    S = BSJ_COMPLEX.(G_cycle)
    interf = UserDefinedInterferometer(to_bsj(fourier_interferometer(M)))
    input  = Input{UserDefinedGramMatrix}(first_modes(M, M), S)
    dist   = full_distribution(input, interf)
    acc, total = zero(BSJ_COMPLEX), zero(BSJ_FLOAT)
    for (o, p) in zip(dist.counts, dist.proba)
        pr = real(p)
        acc += pr * fourier_phase_factor(o.state, M)
        total += pr
    end
    return acc / total          # re-normalise against floating-point drift
end

"""
    fourier_bargmann_estimate(G_cycle::AbstractMatrix{<:Complex}, N::Integer)
        -> ComplexF64

Physical (finite-N) Fourier estimator of `B_π`: draw `N` i.i.d. M-photon
outcomes of `G_cycle = G[π, π]` through `fourier_interferometer(M)` (via
`one_shot_sample`, i.e. BosonSampling.jl's Householder sampler) and return

    B̂_π = (1/N) Σ_samples exp((2πi/M)·f(S)).

Unbiased with `Var(B̂) = (1 − |B_π|²)/N` (each term is unit-modulus);
Novo's Proposition gives `O(ε⁻² ln δ⁻¹)` samples for additive error ε.
The BSJ sampler draws from the global RNG; seed with `Random.seed!` for
reproducibility. `threaded` parallelises the `N` draws inside BSJ's batch
sampler (independent task-local RNGs); it defaults to `true` whenever Julia
was started with more than one thread (`Threads.nthreads() > 1`), so the
estimator uses all available cores by default — pass `threaded=false` for the
exact serial RNG stream.
"""
function fourier_bargmann_estimate(G_cycle::AbstractMatrix{<:Complex}, N::Integer;
                                   threaded::Bool = Threads.nthreads() > 1)
    M = size(G_cycle, 1)
    @assert size(G_cycle, 2) == M "G_cycle must be square"
    @assert N > 0
    samples = one_shot_sample(G_cycle, fourier_interferometer(M), N; threaded=threaded)
    return sum(fourier_phase_factor(S, M) for S in samples) / N
end

# ----------------------------------------------------------------------------
# HOM — the M = 2 special case (per-pair physical magnitude estimator).
#
# The general Fourier estimator above measures B_π at any order M through the
# F̂† interferometer. At M = 2 the interferometer is the 50:50 beamsplitter
# (`fourier_interferometer(2) == HOM_U_PHYSICS`) and B_π = |G_{ij}|² is real,
# so a Hong–Ou–Mandel coincidence count suffices. This explicit two-mode
# estimator is kept as the simplest physical magnitude estimator and as the
# comparison baseline for the general method; it shares the BSJ boundary
# (`to_bsj`, `_run_experiments`, precision pins) of `boson_sampling_integration.jl`.
# ----------------------------------------------------------------------------

"""50:50 beamsplitter in the physics convention (`b†_out = U_HOM · a†_in`).
Since this matrix is symmetric, it equals its own transpose and so its
BosonSampling.jl-convention counterpart is itself; it equals
`fourier_interferometer(2)`."""
const HOM_U_PHYSICS = BSJ_COMPLEX[1.0  1.0;
                                  1.0 -1.0] ./ sqrt(2)

"""
    hom_probabilities(s::Number) -> (p20, p11, p02)

Analytical HOM output probabilities for two single-photon inputs with
`⟨ψ_1|ψ_2⟩ = s` through a 50:50 beamsplitter:

    P((2,0)) = P((0,2)) = (1 + |s|²) / 4,    P((1,1)) = (1 − |s|²) / 2.
"""
function hom_probabilities(s::Number)
    v = abs2(s)
    return ((1 + v) / 4, (1 - v) / 2, (1 + v) / 4)
end

"""
    hom_sample_coincidences(s::Number, N::Integer) -> Int

Generate `N` i.i.d. HOM outcomes for `s = G_{ij}` via BosonSampling.jl's
Householder-based partial-distinguishability sampler (`sample_multiple`),
which is distribution-correct for complex Hermitian `S` since
BosonSampling.jl commit `edbd641`. Returns the number of coincidence
events (output occupation `[1, 1]`).

The BSJ sampler draws from the global RNG; seed with `Random.seed!` for
reproducibility.
"""
function hom_sample_coincidences(s::Number, N::Integer)
    S = BSJ_COMPLEX[1.0       s;
                    conj(s)   1.0]
    interf = UserDefinedInterferometer(to_bsj(HOM_U_PHYSICS))
    input  = Input{UserDefinedGramMatrix}(first_modes(2, 2), S)
    samples = sample_multiple(input, interf, N)
    return count(o -> o == [1, 1], samples)
end

"""
    estimate_B_ij(s::Number, N::Integer) -> Float64

Unbiased HOM estimator of `|s|² = B_{ij}` (the M = 2 Bargmann invariant):

    B̂_{ij} = 1 - 2 · N_{coinc} / N,     Var(B̂) = (1 - |s|⁴) / N.
"""
function estimate_B_ij(s::Number, N::Integer)
    n_coinc = hom_sample_coincidences(s, N)
    return 1 - 2 * n_coinc / N
end

"""
    hom_tomography_magnitudes(G::AbstractMatrix{<:Complex}, N::Integer) -> Matrix{Float64}

For each off-diagonal pair `(i, j)` of an `n × n` pure-state Gram matrix `G`,
run an HOM experiment with `N` samples and return the symmetric matrix `B̂`
of estimated `|G_{ij}|²`. Diagonal entries are 1.

This is the magnitude half of the tomography pipeline; downstream code
composes `sqrt.(max.(B̂, 0.0))` with the phase information of higher-order
invariants (the general Fourier estimator above) to fully reconstruct `G`.

The `C(n,2)` per-pair HOM experiments are independent; `threaded` runs them
concurrently (see `_run_experiments`). It defaults to `true` whenever Julia was
started with more than one thread (`Threads.nthreads() > 1`), so all available
cores are used by default — pass `threaded=false` for the exact serial stream.
"""
function hom_tomography_magnitudes(G::AbstractMatrix{<:Complex}, N::Integer;
                                   threaded::Bool = Threads.nthreads() > 1)
    n = size(G, 1)
    @assert size(G, 2) == n "G must be square"
    pairs = [(i, j) for i in 1:n for j in (i+1):n]
    vals = _run_experiments(threaded,
        [() -> estimate_B_ij(G[i, j], N) for (i, j) in pairs])
    B̂ = zeros(BSJ_FLOAT, n, n)
    for i in 1:n; B̂[i, i] = 1.0; end
    for (p, (i, j)) in enumerate(pairs)
        B̂[i, j] = vals[p]
        B̂[j, i] = vals[p]
    end
    return B̂
end
