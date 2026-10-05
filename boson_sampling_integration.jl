"""
Integration with BosonSampling.jl: the package boundary and the generic
sampling primitives built directly on it — precision pin, the physics-to-BSJ
unitary transpose adapter, feature detection, the probability wrapper, the
across-experiment parallel runner, the occupation-sample draw
(`one_shot_sample`), and the Haar-unitary generator (`random_haar_unitary`).
The method-specific *estimators* that consume these primitives live in their
own files: the HOM / Fourier multivariate-trace estimator in
`fourier_estimator.jl` (HOM is its `M = 2` case) and the correlator one-shot
estimator in `one_shot_estimator.jl`; the end-to-end `physical_tomography_*`
drivers that wire an estimator to the protocol live in `physical_tomography.jl`.

## Convention (critical)

Physics convention (notes + this project): `b†_{out, i} = Σ_j U_{ij} a†_{in, j}`
— row = output, col = input.

BosonSampling.jl's `process_probability_partial` / `scattering_matrix`
use the TRANSPOSED convention (`U[i, j]` = input `i` → output `j`). So
every call into the package passes `to_bsj(U_phys) = transpose(U_phys)`,
NEVER `U_phys` and NEVER `adjoint(U_phys)`. (Adjoint adds a spurious
complex conjugation; transpose is correct because the package never
applies `conj` to U itself.) The Gram-matrix convention `S[i,j] = ⟨ψ_i|ψ_j⟩`
matches between the two. See `CLAUDE.md` § "Interferometer unitary" for
the long-form discussion and the test
`@testset "BosonSampling: unitary convention is transpose-of-physics"`
for the lock-down.

## Required fixes

Three BosonSampling.jl commits are required — `bda901d` (Clifford bias
fix), `f5c9bbe` (partial-distinguishability sampler), and `91b661a`
(`MixedDensityMatrices` input type for mixed-state sampling, the current
head of the `samplers` branch). `require_fixed_samplers()` feature-detects
all three.
"""

using LinearAlgebra
using Random
using BosonSampling

"""
    BSJ_FLOAT, BSJ_COMPLEX

The floating-point precision pinned by BosonSampling.jl. The package evaluates
permanents (`ryser`) and output probabilities exclusively in `Float64` /
`ComplexF64`, so every Gram matrix and unitary crossing the `to_bsj` boundary
is converted to `BSJ_COMPLEX`, and every probability or photon-count-derived
estimator returns `BSJ_FLOAT`.

These are NOT a stylistic Float64 default: high-precision element types
(`BigFloat`, `Double64`, …) propagate freely through the *pure-math* modules
(`gram_matrix.jl`, `spanning_tree.jl`, `estimators.jl`, `projection.jl`,
`reconstruction.jl`, `protocol.jl`, which all derive their working type from
their inputs), but any quantity that is *physically sampled* collapses to this
precision — a property of the external dependency, not a project convention.
Pinned here, in the one module that owns the BSJ boundary, so the constraint is
explicit and centralized rather than scattered as bare `ComplexF64` literals.
"""
const BSJ_FLOAT = Float64
const BSJ_COMPLEX = Complex{BSJ_FLOAT}

"""Cyclic 3-mode shift, used as a convention discriminator. Physically maps
input 1 → output 3, input 2 → output 1, input 3 → output 2. Non-symmetric,
real-orthogonal."""
const U_SHIFT_PHYSICS = BSJ_COMPLEX[0 0 1;
                                    1 0 0;
                                    0 1 0]

"""
    to_bsj(U_phys)

Convert a unitary from the physics convention (`out = U_phys · in`) to the
convention expected by BosonSampling.jl's `process_probability_partial` /
`scattering_matrix` (`U[i, j]` = amplitude for input mode i → output mode j).
The conversion is a plain **transpose**, not an adjoint — see module
docstring for the reason.
"""
to_bsj(U_phys::AbstractMatrix) = Matrix(transpose(U_phys))

"""
    _run_experiments(threaded::Bool, thunks::AbstractVector) -> Vector

Run a list of zero-argument experiment closures and collect their results in
order. The boson-sampling experiments behind these closures are mutually
independent (one interferometer + sample batch each), so with `threaded=true`
each runs in its own `Threads.@spawn` task. Every spawned task gets an
independent, deterministically seeded task-local RNG, so the collected results
are bit-reproducible under a top-level `Random.seed!` and the caller's own RNG
state is left untouched (verified in `tests.jl`). With `threaded=false`
(default) the closures run serially, preserving the exact pre-threading RNG
stream. Use this for *across-experiment* parallelism; for *within-experiment*
parallelism (one big sample batch) pass `threaded=true` to the sampler instead.
"""
function _run_experiments(threaded::Bool, thunks::AbstractVector)
    threaded || return [f() for f in thunks]
    tasks = [Threads.@spawn f() for f in thunks]
    return fetch.(tasks)
end

"""
    require_fixed_samplers()

Throw an informative error unless the loaded BosonSampling.jl has the
corrected Clifford sampler (commit `bda901d`), the partial-distinguishability
sampler (commit `f5c9bbe`), and the mixed-state input type (commit `91b661a`).
Returns `(; clifford_fix=true, partial_dist=true, mixed_state=true)` on success.
"""
function require_fixed_samplers()
    has_clifford = isdefined(BosonSampling, :corrected_clifford_algorithm)
    has_partdist = isdefined(BosonSampling, :PartialDistinguishabilityModel)
    has_mixed    = isdefined(BosonSampling, :MixedDensityMatrices)
    if !(has_clifford && has_partdist && has_mixed)
        missing_items = String[]
        has_clifford || push!(missing_items,
            "corrected_clifford_algorithm (commit bda901d: \"Fix Clifford algorithm bias\")")
        has_partdist || push!(missing_items,
            "PartialDistinguishabilityModel (commit f5c9bbe: \"partial-distinguishability sampler\")")
        has_mixed || push!(missing_items,
            "MixedDensityMatrices (commit 91b661a: \"Add MixedDensityMatrices input type for mixed-state sampling\")")
        error("""
        BosonSampling.jl is missing the following required fix(es):
          - $(join(missing_items, "\n          - "))
        Check out a branch / commit that contains all three; the 'samplers'
        branch of https://github.com/benoitseron/BosonSampling.jl is the
        canonical source (head must be at 91b661a or later).
        """)
    end
    return (; clifford_fix=true, partial_dist=true, mixed_state=true)
end

"""
    bsj_physics_probability(U_phys, S, input, output) -> Float64

Thin wrapper that sends the physics-convention unitary `U_phys` through
BosonSampling.jl's `process_probability_partial` with the correct transpose.
Returns the real part of the probability (imaginary part should be zero up to
floating-point error).
"""
function bsj_physics_probability(U_phys::AbstractMatrix,
                                 S::AbstractMatrix,
                                 input::AbstractVector{<:Integer},
                                 output::AbstractVector{<:Integer})
    U_bsj = to_bsj(U_phys)
    return real(process_probability_partial(U_bsj, Matrix{BSJ_COMPLEX}(S),
                                            collect(input), collect(output)))
end

"""
    one_shot_sample(G::AbstractMatrix{<:Complex}, U_phys::AbstractMatrix,
                    N::Integer; threaded::Bool=false)
        -> Vector{Vector{Int}}

Generate `N` i.i.d. mode-occupation outputs from a single boson-sampling
experiment with pure-state Gram matrix `G` (size `n × n`), input state
`|1^n, 0^{m−n}⟩`, and interferometer `U_phys` (physics convention,
`m × m` with `m ≥ n`).

Generic sampling primitive of the BSJ boundary — the shared draw used by both
estimator families: the correlator route (`one_shot_estimator.jl`) and the
Fourier route (`fourier_estimator.jl`, via `fourier_bargmann_estimate`). The
`one_shot` in the name is "one experiment of N samples", not the correlator
method specifically.

Uses BosonSampling.jl's Householder-based partial-distinguishability
sampler (`sample_multiple`; feature-detected by `require_fixed_samplers`).
It decomposes `G = V V†`, embeds the problem into an `n·m`-mode bosonic
sampling problem, and runs the corrected Clifford sampler per draw — exact,
and does NOT enumerate the full `C(m+n-1, n)` output distribution, so it
remains usable when `full_distribution` becomes infeasible.

Convention: `interf.U` is passed as `to_bsj(U_phys)` so the Clifford
sampler's internal transpose hands back the physics-convention result. The
BSJ sampler draws from the global RNG; seed with `Random.seed!` for
reproducibility.

Correctness is locked by `test/test_householder_distribution.jl` in
BosonSampling.jl (Bosonic/distinguishable limits and HOM dip all match
`full_distribution` to statistical noise).
"""
function one_shot_sample(G::AbstractMatrix{<:Complex},
                         U_phys::AbstractMatrix,
                         N::Integer;
                         threaded::Bool=false)
    n = size(G, 1)
    m = size(U_phys, 1)
    @assert m ≥ n "interferometer must have at least n modes"
    S = BSJ_COMPLEX.(G)
    interf = UserDefinedInterferometer(to_bsj(U_phys))
    input  = Input{UserDefinedGramMatrix}(first_modes(n, m), S)
    return sample_multiple(input, interf, N; threaded=threaded)
end

"""
    random_haar_unitary(m::Integer; rng=Random.default_rng()) -> Matrix{ComplexF64}

Draw an `m × m` Haar-random unitary. Uses QR with phase fix-up. (Duplicates
`BosonSampling.rand_haar` but keeps an `rng` kwarg the package lacks.)
"""
function random_haar_unitary(m::Integer; rng=Random.default_rng())
    Z = randn(rng, BSJ_COMPLEX, m, m) / sqrt(2)
    Q, R = qr(Z)
    Λ = Diagonal(sign.(diag(R)))   # complex signs = phases
    return Matrix(Q) * Λ
end
