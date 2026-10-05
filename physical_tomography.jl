"""
End-to-end physical Gram-matrix tomography drivers.

This file holds the *drivers* that wire a Bargmann-invariant estimator to the
supplier-agnostic protocol of `protocol.jl` (δ-cutoff → MDST → assembly →
projection) and run the whole pipeline from a true Gram matrix `G` to a
reconstructed `Ĝ`. The estimators themselves live one layer down:

- correlator (one-shot) order-2/order-3 estimator → `one_shot_estimator.jl`;
- Fourier multivariate-trace estimator (+ HOM, its M=2 case) → `fourier_estimator.jl`;
- the BSJ sampling primitive `one_shot_sample` → `boson_sampling_integration.jl`.

Kept in its own file (included last) so the cross-method drivers — the hybrid
correlator+Fourier route and the unified `physical_tomography` front door — can
reference *both* estimators top-down without either estimator file depending on
the other. Each `physical_tomography_*` driver below uses `G` only to simulate
the device's samples; the reconstruction itself never inspects `G` (genuine
black box), and the returned `Ĝ` matches `G` only up to the MDST diagonal gauge
(compare against `regauge_to_tree(G, result.T)`).

Six drivers, in dependency order:

1. `physical_tomography_from_samples`    — correlator reconstruction on
   pre-generated samples (the shared core of the one-shot/hybrid routes);
2. `physical_tomography_from_experiment` — LAB-FACING wrapper of (1): validates
   real device samples + a physics-convention `U`, carries the assumptions warning;
3. `physical_tomography_one_shot`        — correlator method, orders ≤ 3 only;
4. `physical_tomography_hybrid`          — correlator + Fourier fallback (cost-optimal);
5. `physical_tomography_fourier`         — Fourier method, any invariant order;
6. `physical_tomography`                 — unified front door, low-order estimator selectable.

# ⚠ Physical assumptions when reconstructing from REAL experimental samples

The `G`-taking drivers below SIMULATE the device, so these assumptions hold by
construction in this codebase. They do NOT hold automatically on hardware. If
you feed real detector samples through `physical_tomography_from_samples` (the
only driver that accepts external samples), the reconstructed `Ĝ` is trustworthy
only insofar as the experiment satisfies:

1. **Perfectly known and perfectly implemented interferometer `U`.** The
   correlator / Fourier inversions invert against the *nominal* `U_phys` the
   caller supplies. Any mismatch between the fabricated/operated unitary and
   `U_phys` (fabrication tolerance, phase miscalibration, drift of the optical
   network) biases *every* Bargmann-invariant estimate. There is no `U`-error
   term anywhere in the protocol or in the Theorem-1 bound — `U` is treated as
   exact and noiseless.

2. **Stationary partial distinguishability — no secular drift during the run.**
   The whole point of the protocol is to recover a *single* Gram matrix
   `G_{ij} = ⟨ψ_i|ψ_j⟩`, assumed constant across the entire acquisition. Slow
   drift of source indistinguishability while data is being recorded
   (temperature, spectral wander, path-length / coupling drift, pump-power
   fluctuation between the per-pair and per-path experiments) means different
   samples report different `G`. The estimators average over them silently and
   `Ĝ` becomes a time-average with no associated error term. The per-pair and
   per-path experiments are also run *sequentially* here, so they are especially
   exposed to drift between them.

These are *systematic* effects. The σ²/N estimator model and the Theorem-1
Frobenius-MSE bound (eq. 40–42) capture only *statistical* finite-sample noise;
they say nothing about (1) or (2). Treat the bound as a best case that assumes a
perfect, stationary device. (Other non-idealities the pure-state model also
excludes — photon loss / sub-unit detection efficiency, multiphoton SPDC
emission, dark counts — break the `ρ² = ρ` pure-state assumption similarly.)
"""

using LinearAlgebra

# ---------------------------------------------------------------------------
# Correlator (one-shot) route
# ---------------------------------------------------------------------------

"""
    physical_tomography_from_samples(n, U_phys, samples; δ::Real=0.0,
                                     long_path_supplier=nothing)
        -> (Ĝ, b2_clamped, B3_hat, T, Γ, n_removed)

The reconstruction half of the correlator method, on pre-generated `samples`
(no boson-sampling). `physical_tomography_one_shot` calls this after drawing
its samples; calling it directly enables N-subsampling — sample once at
`N_max`, then pass `samples[1:N]` here for each N to study scaling without
re-computing the output distribution.

Steps:
1. order-2 inversion (the correlator box of
   `Accessing_relation_information.tex`) → `|G_{kl}|²` for every pair;
2. protocol steps 2–3 (`protocol.jl`): δ-cutoff graph `Γ`, MDST `T`;
3. for each surviving non-tree edge, supply `B_{π(i,j)}`. A triangle path
   (order 3) comes from the order-3 inversion (Robbio Eq. 29) on the SAME
   samples. A longer path (order > 3) is beyond the correlator method's
   reach (orders ≤ 3 ONLY, manuscript) and is delegated to the optional
   `long_path_supplier(i, j, path) -> B_{π(i,j)}::Complex`. With the default
   `long_path_supplier = nothing` (the pure correlator route) a path of
   order > 3 ERRORS, so this route reconstructs only graphs whose MDST has
   hop diameter ≤ 2. The Fourier method supplies arbitrary orders; wiring it
   in as `long_path_supplier` is exactly `physical_tomography_hybrid`;
4. protocol steps 5–6: assembly + Higham projection (`reconstruct_gram`).

The returned `b2_clamped` is the magnitude matrix actually used downstream
(clamped to `[1e-8, 1]`), NOT the raw unbiased order-2 estimate. `B3_hat`
is an empty Dict when no triangle path survives.

!!! warning "Real hardware samples"
    This is the only driver that accepts externally generated `samples`. When
    they come from a physical device rather than this codebase's simulator, the
    inversion assumes `U_phys` is *exactly* the implemented unitary and that the
    underlying Gram matrix is *stationary* over the whole acquisition (no secular
    drift of partial distinguishability between/within experiments). Neither is
    checked, and neither is captured by the Theorem-1 bound — see the
    "Physical assumptions" block at the top of this file.
"""
function physical_tomography_from_samples(n::Integer,
                                          U_phys::AbstractMatrix,
                                          samples::AbstractVector;
                                          δ::Real=0.0,
                                          long_path_supplier=nothing)
    # Order-2 linear inversion. Keep the raw (possibly-negative) estimate:
    # it is the unbiased input to the order-3 subtraction (Eq. 29's
    # 2-Bargmann term). Clamp a separate copy only where a non-negative
    # magnitude is required (graph weights, √ in assembly); the 1e-8 floor
    # (not 0) keeps the MDST γ-weights finite when a noisy estimate
    # undershoots.
    b2_hat, _, _, _ = one_shot_bargmann_order_2(U_phys, samples, n)
    b2_raw = real.(b2_hat)                       # unbiased
    b2_pos = clamp.(b2_raw, 1e-8, 1.0)           # for δ-cutoff + MDST + sqrt
    for k in 1:n; b2_pos[k, k] = 1.0; b2_raw[k, k] = 1.0; end

    # Protocol steps 2–3 on the estimated magnitudes (n(δ) is recomputed and
    # returned by reconstruct_gram below).
    Γ, _ = protocol_graph(b2_pos, δ)
    T = protocol_tree(Γ, b2_pos)

    # Step 4: supply B_{π(i,j)} for every surviving non-tree edge. A triangle
    # path (order 3) is served natively by the order-3 inversion; a longer
    # path (order > 3) is delegated to `long_path_supplier`, or errors when
    # none is given (pure correlator route, orders ≤ 3 only).
    reqs = required_paths(Γ, T)
    needs_order3 = any(r -> length(r[2]) == 3, reqs)
    B3_hat = Dict{NTuple{3, Int}, BSJ_COMPLEX}()
    Bpath  = Dict{Tuple{Int, Int}, BSJ_COMPLEX}()
    if needs_order3
        # Order-3 inversion (once, on the SAME samples) uses the RAW
        # (unbiased) b2 so the Eq. (29) subtraction doesn't inherit a
        # positive bias from clamping.
        B3_hat, _, _ = one_shot_bargmann_order_3(U_phys, samples, n, b2_raw)
    end
    for ((i, j), path) in reqs
        # length(path) == 2 cannot occur for a non-tree edge.
        if length(path) == 3
            Bpath[(i, j)] = bargmann_invariant_for_triple(
                B3_hat, path[1], path[2], path[3])
        elseif long_path_supplier !== nothing
            Bpath[(i, j)] = long_path_supplier(i, j, path)
        else
            error(
                "correlator method supplies Bargmann invariants of order ≤ 3 " *
                "only (manuscript, Accessing relation information); required " *
                "tree path $(path) for edge ($i,$j) has order $(length(path)) " *
                "> 3. Higher orders need the Fourier method " *
                "(physical_tomography_fourier in fourier_estimator.jl), or " *
                "pass `long_path_supplier` (see `physical_tomography_hybrid`).")
        end
    end

    # Protocol steps 5–6 (assembly from eq:gram1/eq:gram2 + projection).
    result = reconstruct_gram(b2_pos, Bpath, δ)
    return (; Ĝ = result.Ĝ, b2_clamped = b2_pos, B3_hat = B3_hat,
            T = result.T, Γ = result.Γ, n_removed = result.n_removed)
end

"""
    physical_tomography_from_experiment(U_phys, samples, n; δ=0.0,
                                        long_path_supplier=nothing,
                                        ack_assumptions=false)
        -> (Ĝ, b2_clamped, B3_hat, T, Γ, n_removed)

LAB-FACING ENTRY POINT — reconstruct the Gram matrix from data you measured on
a real device, rather than from a simulated `G`. This is the only entry point
intended to be called with samples that did NOT come from this codebase's
simulator; it is a thin, validated wrapper over `physical_tomography_from_samples`
(the correlator route — order-2 + order-3 linear inversions; orders ≤ 3, so the
MDST must have hop diameter ≤ 2, see that function).

# Inputs you must provide
- `U_phys :: AbstractMatrix` — the `m × m` interferometer unitary in the
  **physics convention** of this project (`bᵢ† = Σⱼ U_{ij} aⱼ†`; row = output,
  column = input). Do NOT pre-transpose for BosonSampling.jl — that adapter
  (`to_bsj`) lives at the package boundary and is applied internally. `m ≥ n`.
- `samples :: AbstractVector` — your `N` measured outcomes. Each element is an
  **occupation-number vector of length `m`**: `samples[t][i]` = number of photons
  detected in output mode `i` on shot `t`, with the `n` input single photons
  injected into the first `n` modes (`Σᵢ samples[t][i] == n`, lossless). This is
  the same layout `one_shot_sample` produces. You are responsible for converting
  your raw click/coincidence records into this format first (drop or post-select
  lossy/dark-count shots so every retained shot has exactly `n` photons).
- `n :: Integer` — the number of single-photon sources / states whose Gram
  matrix is being reconstructed.

!!! warning "This protocol trusts your device — read before relying on Ĝ"
    Feeding real samples here assumes, *with no check and no error term in the
    Theorem-1 bound (eq. 40–42)*:
    1. **`U_phys` is exactly the implemented unitary.** Fabrication tolerance,
       phase miscalibration, or drift of the optical network biases every
       Bargmann-invariant estimate — the inversion is performed against the
       nominal `U_phys` you pass in.
    2. **Partial distinguishability is stationary across the whole run** — no
       secular drift of source indistinguishability (temperature, spectral
       wander, path-length / coupling drift) while the `N` shots are recorded.
       Drift makes `Ĝ` a silent time-average of a moving Gram matrix.
    These are *systematic* effects; the σ²/N model and the bound capture only
    *statistical* finite-sample noise. See the "Physical assumptions" block at
    the top of this file. Pass `ack_assumptions=true` to silence this warning
    once you have read it.
"""
function physical_tomography_from_experiment(U_phys::AbstractMatrix,
                                             samples::AbstractVector,
                                             n::Integer;
                                             δ::Real=0.0,
                                             long_path_supplier=nothing,
                                             ack_assumptions::Bool=false)
    require_fixed_samplers()
    m = size(U_phys, 1)
    @assert size(U_phys, 2) == m "U_phys must be square"
    @assert m ≥ n "interferometer must have at least n modes (m ≥ n)"
    @assert !isempty(samples) "no samples provided"
    # Validate the occupation-sample format: length-m, non-negative integer
    # counts, exactly n photons per shot (lossless). These are the assumptions
    # the correlator inversions rely on — fail loudly rather than silently
    # mis-inverting on mis-shaped lab data.
    let s1 = first(samples)
        @assert length(s1) == m "each sample must be an occupation vector of \
            length m = $m (got $(length(s1)))"
    end
    for (t, s) in enumerate(samples)
        @assert length(s) == m "sample $t has length $(length(s)), expected m = $m"
        @assert all(x -> x ≥ 0 && isinteger(x), s) "sample $t has non-integer or \
            negative occupations; expected photon counts"
        @assert sum(s) == n "sample $t has $(sum(s)) photons, expected n = $n \
            (drop/post-select lossy or dark-count shots before calling)"
    end

    ack_assumptions || @warn """
        physical_tomography_from_experiment: reconstructing a Gram matrix from \
        real samples assumes (1) U_phys is EXACTLY the implemented unitary and \
        (2) partial distinguishability did not drift during acquisition. Neither \
        is checked or bounded. See the docstring / the top-of-file "Physical \
        assumptions" block. Pass ack_assumptions=true to silence.""" maxlog=1

    return physical_tomography_from_samples(n, U_phys, samples;
                                            δ=δ, long_path_supplier=long_path_supplier)
end

"""
    physical_tomography_one_shot(G_true, U_phys, N; δ::Real=0.0)
        -> (Ĝ, b2_clamped, B3_hat, T, Γ, n_removed, samples)

End-to-end correlator-method reconstruction of a pure-state Gram matrix
from a SINGLE experiment: `N` samples through the fixed interferometer
`U_phys`, then `physical_tomography_from_samples` (order-2 + order-3
linear inversions on the same samples, supplier-agnostic protocol from
`protocol.jl`).

The correlator method supplies Bargmann invariants of order ≤ 3 only
(`Accessing_relation_information.tex`), so the reconstruction works only
when the MDST has hop diameter ≤ 2 (every non-tree cycle is a triangle);
it errors otherwise — see `physical_tomography_from_samples`.

`G_true` is used only to simulate the sampling — the reconstruction
itself never sees it (genuine black box). The output `Ĝ` matches `G`
up to the MDST-induced diagonal gauge; use `regauge_to_tree(G,
result.T)` to compare. Sampling draws from the global RNG; seed with
`Random.seed!` for reproducibility. `threaded` parallelises the `N` draws
across threads (independent task-local RNGs; ≈3–4×, statistically identical,
stream differs from the serial draw); it defaults to `true` whenever Julia was
started with more than one thread (`Threads.nthreads() > 1`), so all available
cores are used by default — pass `threaded=false` for the exact serial stream.
"""
function physical_tomography_one_shot(G_true::AbstractMatrix,
                                      U_phys::AbstractMatrix,
                                      N::Integer;
                                      δ::Real=0.0,
                                      threaded::Bool = Threads.nthreads() > 1)
    require_fixed_samplers()
    n = size(G_true, 1)
    @assert size(G_true, 2) == n
    @assert size(U_phys, 1) == size(U_phys, 2) "U must be square"
    @assert size(U_phys, 1) >= n "need m ≥ n modes"

    # ONE experiment, N samples; everything downstream is `samples`-only, so
    # the reconstruction is shared with `physical_tomography_from_samples`.
    # `threaded=true` parallelises the N draws across threads inside BSJ's
    # batch sampler (each draw uses an independent task-local RNG).
    samples = one_shot_sample(G_true, U_phys, N; threaded=threaded)
    result  = physical_tomography_from_samples(n, U_phys, samples; δ=δ)
    return (; result..., samples = samples)
end

"""
    physical_tomography_hybrid(G_true, U_phys, N; δ::Real=0.0,
                               N_fourier::Integer=N,
                               threaded::Bool=Threads.nthreads()>1)
        -> (; Ĝ, b2_clamped, B3_hat, T, Γ, n_removed, samples, fourier_edges)

The manuscript's cost-optimal estimation strategy: use the single-experiment
**correlator method** for everything it reaches (orders ≤ 3 — all order-2
magnitudes that build the graph/MDST, and all triangle paths), and fall back
to the **Fourier method** ONLY for the residual tree paths of order > 3.

This realises the intent behind the two estimator boxes of
`Accessing_relation_information.tex`: the correlator box ("all Bargmann
invariants of order k ≤ 3" from one experiment) is the workhorse, and the
Fourier box ("Measuring the Bargmann invariant Bπ", one experiment per
invariant at any order) is the last resort. Compared with
`physical_tomography_fourier` — which spends one Fourier experiment per pair
*and* per path (`C(n,2) + #paths` experiments) — the hybrid spends one
correlator experiment plus one Fourier experiment per order-> 3 tree path. On
any graph whose MDST has hop diameter ≤ 2 it makes ZERO Fourier calls and is
bit-identical to `physical_tomography_one_shot`.

- `G_true` is used only to simulate the device's samples (the correlator
  experiment through `U_phys`, and each fallback Fourier experiment on the
  sub-Gram `G_true[path, path]`); the reconstruction itself never inspects
  it. `N` samples for the correlator run, `N_fourier` per Fourier path (the
  high-order paths are the noisiest, so they get their own budget; default
  `N_fourier = N`).
- `Ĝ` matches `G` up to the MDST diagonal gauge — compare against
  `regauge_to_tree(G, result.T)`.
- `fourier_edges :: Vector{Tuple{Int,Int}}` lists exactly the non-tree edges
  whose `B_{π}` came from the Fourier fallback (empty ⇔ pure correlator run).
  When non-empty, a single summary `@info` reports how many order-> 3 paths
  escalated to the Fourier method and at which orders (silent otherwise).
- Sampling draws from the global RNG; seed with `Random.seed!` for
  reproducibility. `threaded` parallelises the correlator's `N` draws
  (passed to `one_shot_sample`); the per-path Fourier fallbacks each draw
  serially (one experiment at a time), matching `physical_tomography_fourier`.
"""
function physical_tomography_hybrid(G_true::AbstractMatrix,
                                    U_phys::AbstractMatrix,
                                    N::Integer;
                                    δ::Real=0.0,
                                    N_fourier::Integer=N,
                                    threaded::Bool = Threads.nthreads() > 1)
    require_fixed_samplers()
    n = size(G_true, 1)
    @assert size(G_true, 2) == n
    @assert size(U_phys, 1) == size(U_phys, 2) "U must be square"
    @assert size(U_phys, 1) >= n "need m ≥ n modes"

    # The SINGLE correlator experiment (orders ≤ 3 for the whole graph).
    samples = one_shot_sample(G_true, U_phys, N; threaded=threaded)

    # Fourier fallback for the rare order-> 3 tree paths: one experiment per
    # path on the true sub-Gram (the "hardware" sees the true states), exactly
    # as `physical_tomography_fourier` does. Record which edges used it.
    fourier_edges  = Tuple{Int, Int}[]
    fourier_orders = Int[]
    supplier = function (i, j, path)
        push!(fourier_edges, (i, j))
        push!(fourier_orders, length(path))
        return fourier_bargmann_estimate(G_true[path, path], N_fourier; threaded=false)
    end

    result = physical_tomography_from_samples(n, U_phys, samples; δ=δ,
                                              long_path_supplier=supplier)

    # Summary notice: the Fourier method is the costly last resort here (one
    # experiment per order-> 3 tree path), so report when (and how much) it was
    # needed. Silent when the single correlator run sufficed (diameter ≤ 2).
    if !isempty(fourier_edges)
        @info "physical_tomography_hybrid: correlator run covered orders ≤ 3; " *
              "$(length(fourier_edges)) order-> 3 tree path(s) escalated to the " *
              "Fourier fallback (orders $(sort(unique(fourier_orders))), " *
              "$N_fourier samples each)." fourier_edges
    end
    return (; result..., samples = samples, fourier_edges = fourier_edges)
end

# ---------------------------------------------------------------------------
# Fourier route (any invariant order)
# ---------------------------------------------------------------------------

"""
    physical_tomography_fourier(G::AbstractMatrix{<:Complex}; N::Integer,
                                δ::Real = 0.0)
        -> (; Ĝ, T, Γ, n_removed, B2, Bpath)

End-to-end Gram-matrix reconstruction with the **Fourier method** (Novo
2026; the "Measuring the Bargmann invariant B_π" box of
`Accessing_relation_information.tex`) as the Bargmann-invariant supplier
for the protocol of `protocol.jl`:

1. every pair `(i, j)`: `B2[i, j] = fourier_bargmann_estimate(G[[i,j],[i,j]], N)`
   — the M = 2 (HOM) invariant, real `≈ |G_{ij}|²`;
2. protocol steps 2–4: δ-cutoff graph `Γ`, MDST `T`, required tree paths;
3. every surviving non-tree edge `(i, j)` with tree path `π`:
   `Bpath[(i, j)] = fourier_bargmann_estimate(G[π, π], N)` — one
   M = length(π) Fourier experiment per path;
4. protocol steps 5–6 (`reconstruct_gram`): assembly + Higham projection.

Because one Fourier experiment measures `B_π` at ANY order M, there is no
order-≤-3 restriction: arbitrary MDST diameters are handled, unlike the
correlator method (`physical_tomography_one_shot`), which errors when the
tree demands an invariant of order > 3. The cost is one experiment (own
interferometer) per pair and per required path, versus the correlator
method's single experiment.

`G` is used only to simulate the device's samples (the hardware "sees"
the true states); the reconstruction consumes `B2`/`Bpath` only. The
output `Ĝ` matches `G` up to the MDST tree gauge — compare against
`regauge_to_tree(G, result.T)`. Sampling draws from the global RNG; seed
with `Random.seed!` for reproducibility. `threaded=true` runs the
independent per-pair and per-path experiments concurrently
(`Threads.@spawn`, independent task-local RNGs; reproducible under a
top-level seed — see `_run_experiments`).
"""
function physical_tomography_fourier(G::AbstractMatrix{<:Complex};
                                     N::Integer, δ::Real = 0.0,
                                     threaded::Bool = Threads.nthreads() > 1)
    require_fixed_samplers()
    n = size(G, 1)
    @assert size(G, 2) == n "G must be square"

    # Step 1 — order-2 invariants for every pair (M = 2 Fourier = HOM). The
    # C(n,2) experiments are independent, so `threaded=true` runs them
    # concurrently via `Threads.@spawn` — each task gets its own task-local
    # RNG (independent, deterministically seeded from the caller's RNG at spawn
    # time, so a top-level `Random.seed!` still reproduces the run; see the
    # estimator's per-call `threaded` for within-experiment threading instead).
    pairs = [(i, j) for i in 1:n for j in (i+1):n]
    # `threaded=false` inside the closures: the parallelism is across
    # experiments (the `_run_experiments` @spawn), so each experiment must draw
    # its own N samples serially — nesting @spawn over @threads would
    # oversubscribe the cores.
    pair_vals = _run_experiments(threaded,
        [() -> fourier_bargmann_estimate(G[[i, j], [i, j]], N; threaded=false) for (i, j) in pairs])
    B2 = Matrix{BSJ_COMPLEX}(I, n, n)
    for (p, (i, j)) in enumerate(pairs)
        B2[i, j] = pair_vals[p]
        B2[j, i] = pair_vals[p]
    end
    # Magnitudes for the protocol: |B̂| ≥ 0 keeps MDST γ-weights finite when
    # a noisy estimate of a weak pair undershoots 0.
    B2mag = abs.(B2)
    for i in 1:n; B2mag[i, i] = 1.0; end

    # Steps 2–4 — δ-cutoff, MDST, measurement list (protocol.jl). n(δ) is
    # recomputed and returned by reconstruct_gram below.
    Γ, _ = protocol_graph(B2mag, δ)
    T = protocol_tree(Γ, B2mag)
    reqs = required_paths(Γ, T)

    # One Fourier experiment per required tree path, at its native order —
    # again independent across paths, so parallelised the same way.
    path_vals = _run_experiments(threaded,
        [() -> fourier_bargmann_estimate(G[path, path], N; threaded=false) for (_, path) in reqs])
    Bpath = Dict{Tuple{Int, Int}, BSJ_COMPLEX}()
    for (p, ((i, j), _)) in enumerate(reqs)
        Bpath[(i, j)] = path_vals[p]
    end

    # Steps 5–6 — assembly + projection. `reconstruct_gram` recomputes Γ and
    # T from (B2mag, δ); both are deterministic, so they coincide with the
    # ones above and `Bpath` covers exactly the required non-tree edges.
    result = reconstruct_gram(B2mag, Bpath, δ)
    return (; Ĝ = result.Ĝ, T = result.T, Γ = result.Γ,
            n_removed = result.n_removed, B2 = B2, Bpath = Bpath)
end

# ---------------------------------------------------------------------------
# Unified front door: selectable low-order (pairwise) estimator
# ---------------------------------------------------------------------------

"""
    physical_tomography(G_true; low_order::Symbol, N::Integer,
                        U_phys=nothing, N_fourier::Integer=N, δ::Real=0.0,
                        threaded::Bool=Threads.nthreads()>1)
        -> (; Ĝ, T, Γ, n_removed, cycles, fourier_edges, low_order)

Unified end-to-end physical Gram-matrix tomography with the **low-order
(pairwise) estimator selectable**. The protocol (`gram-estimation.tex`) is
agnostic to how each Bargmann invariant is measured, and there is no reason
the order-2 pairwise invariants must use the same method as the higher-order
cycles. `low_order` picks how the pairwise `B_{ij}` — which build the graph
and MDST — are obtained:

- `:one_shot` (needs `U_phys`): ONE correlator experiment supplies all
  pairwise `B_{ij}` AND all order-3 cycles from the same samples; the Fourier
  method is the last resort, used only for cycles of order > 3. This is the
  practical recommendation (always use one-shot for the low-order invariants;
  generic states never need anything beyond order 3). Delegates to
  `physical_tomography_hybrid`.
- `:fourier`: one Fourier experiment per pair (M = 2) for the pairwise
  `B_{ij}`, and one per required cycle at its native order — no `U_phys`
  needed. Delegates to `physical_tomography_fourier`.

Returns a single concrete (type-stable) named tuple, identical in shape for
both modes: `Ĝ`, the tree `T`, graph `Γ`, `n_removed`, the phase-1 `cycles`
plan (`(; edge, path, order)`, as `plan_required_cycles`), the
`fourier_edges` whose `B_π` came from the Fourier method (every non-tree edge
when `low_order=:fourier`; only the order-> 3 ones when `:one_shot`), and the
selected `low_order`. The full method-specific outputs (`samples` /
`b2_clamped` for one-shot, `B2` / `Bpath` for Fourier) differ in type between
modes, so they are NOT returned here — call `physical_tomography_hybrid` /
`physical_tomography_fourier` directly if you need them.

For the genuinely two-phase experimental workflow — estimate the pairwise
invariants, STOP, then go measure specific cycles in a separate run — use
`plan_required_cycles` + `reconstruct_gram` directly instead of this
all-in-one driver (which samples everything itself).
"""
function physical_tomography(G_true::AbstractMatrix; low_order::Symbol,
                             N::Integer, U_phys=nothing,
                             N_fourier::Integer=N, δ::Real=0.0,
                             threaded::Bool = Threads.nthreads() > 1)
    if low_order === :one_shot
        U_phys === nothing &&
            error("physical_tomography(low_order=:one_shot) needs the " *
                  "interferometer U_phys for the single correlator experiment")
        r = physical_tomography_hybrid(G_true, U_phys, N; δ=δ,
                                       N_fourier=N_fourier, threaded=threaded)
        cycles = [(; edge=e, path=p, order=length(p))
                  for (e, p) in required_paths(r.Γ, r.T)]
        return (; r.Ĝ, r.T, r.Γ, r.n_removed, cycles,
                fourier_edges = r.fourier_edges, low_order)
    elseif low_order === :fourier
        r = physical_tomography_fourier(G_true; N=N, δ=δ, threaded=threaded)
        reqs = required_paths(r.Γ, r.T)
        cycles = [(; edge=e, path=p, order=length(p)) for (e, p) in reqs]
        return (; r.Ĝ, r.T, r.Γ, r.n_removed, cycles,
                fourier_edges = [e for (e, _) in reqs], low_order)
    else
        error("physical_tomography: low_order must be :one_shot or " *
              ":fourier, got :$low_order")
    end
end
