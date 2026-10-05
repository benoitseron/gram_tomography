"""
One-shot (correlator-method) Bargmann-invariant estimator.

Implements the "Measuring all Bargmann invariants of order k ≤ 3" box of
`notes/gtomo_new/Sections/Accessing_relation_information.tex`: a single
boson-sampling experiment through one interferometer `U`, `N` samples,
linear inversion → every pairwise `b_{kl} = |G_{kl}|²` and every triangle
`B_{rst} = Tr(ρ_r ρ_s ρ_t)`. Orders 2 and 3 ONLY — higher orders need the
Fourier method (`fourier_estimator.jl`).

This file is the *estimator* (the linear-inversion machinery + the order-3
output decoder `bargmann_invariant_for_triple`); the end-to-end driver that
samples through `U`, inverts, and runs the protocol to a reconstructed `Ĝ`
(`physical_tomography_one_shot` / `_from_samples` / `_hybrid`) lives in
`physical_tomography.jl`.

- Order 2 (`one_shot_bargmann_order_2`): solves
      `A(U)·b = c − c^{dist}`,   A(U)_{ij;kl} = U_{ik}·U_{jl}·conj(U_{il})·conj(U_{jk}),
  with `c_{ij} = ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩` and
  `c^{dist}_{ij} = −Σ_k |U_{ik}|²|U_{jk}|²` (the distinguishable
  baseline one must subtract — missing from the notes as literally
  stated). Appendix A of `notes/Gram_Tomography.pdf`.
- Order 3 (`one_shot_bargmann_order_3`): same samples, Eq. (29) of
  Marco Robbio's "Correlator-based witness of indistinguishability"
  (2025-09-10, in `notes/literature/`). Because `⟨n_a n_b n_c⟩ ∈ ℝ`
  and `Tr(ρ_s ρ_t ρ_r) = conj(Tr(ρ_s ρ_r ρ_t))`, the 3-Bargmann block
  reduces to a REAL linear system in `(Re B_{rst}, Im B_{rst})`.

Convention: `U` is physics-convention (row = output, col = input).
Sampling routes through `to_bsj(U) = transpose(U)` at the
BosonSampling.jl boundary; see `boson_sampling_integration.jl` and
`CLAUDE.md` § "Interferometer unitary — U vs Uᵀ".
"""

using LinearAlgebra
using Random
using BosonSampling

"""
    build_A_matrix_order_2(U, n) -> (A, out_pairs, in_pairs)

Build the `(m² − m) × (n² − n)` matrix `A(U)` (`eq:A(U)` in
`notes/gtomo_new/Sections/condition-number.tex`). Rows are indexed by ordered
pairs `(i, j)` of distinct **output** modes; columns by ordered pairs `(k, l)`
of distinct **input** photons.
"""
function build_A_matrix_order_2(U::AbstractMatrix, n::Integer)
    m = size(U, 1)
    out_pairs = [(i, j) for i in 1:m for j in 1:m if i != j]
    in_pairs  = [(k, l) for k in 1:n for l in 1:n if k != l]
    n_rows = length(out_pairs)
    n_cols = length(in_pairs)
    A = zeros(eltype(U), n_rows, n_cols)
    @inbounds for (a, (i, j)) in enumerate(out_pairs)
        for (b, (k, l)) in enumerate(in_pairs)
            A[a, b] = U[i, k] * U[j, l] * conj(U[i, l]) * conj(U[j, k])
        end
    end
    return A, out_pairs, in_pairs
end

"""
    empirical_mode_correlators(samples, m, out_pairs) -> Vector{Float64}

Given a vector of `N` mode-occupation vectors `samples[r] ∈ ℤ^m`,
return the empirical connected correlator vector indexed by `out_pairs`:

    c_{ij} = (1/N) ∑_r n_i^{(r)} n_j^{(r)}  −  ⟨n_i⟩ ⟨n_j⟩.

Single pass over `samples`: each occupation vector has only `n ≪ m` occupied
modes, so we accumulate the full `m × m` raw-moment matrix `⟨n_i n_j⟩` touching
only occupied-mode pairs, then read off `out_pairs`. This is `O(N·(m + n²))`
versus the naive `O(N·m²)` of re-scanning `samples` once per output pair.
"""
function empirical_mode_correlators(samples::AbstractVector,
                                    m::Integer,
                                    out_pairs::AbstractVector)
    N = length(samples)
    @assert N > 0
    mean_n = zeros(BSJ_FLOAT, m)
    moment = zeros(BSJ_FLOAT, m, m)        # ⟨n_i n_j⟩ accumulator (raw sums)
    occ = Vector{Int}(undef, m)            # reused occupied-mode buffer
    @inbounds for s in samples
        k = 0
        for i in 1:m
            si = s[i]
            if si != 0
                k += 1; occ[k] = i
                mean_n[i] += si
            end
        end
        for x in 1:k
            i = occ[x]; si = s[i]
            for y in 1:k
                j = occ[y]
                moment[i, j] += si * s[j]
            end
        end
    end
    mean_n ./= N

    c = Vector{BSJ_FLOAT}(undef, length(out_pairs))
    @inbounds for (a, (i, j)) in enumerate(out_pairs)
        c[a] = moment[i, j] / N - mean_n[i] * mean_n[j]
    end
    return c
end

"""
    distinguishable_correlator_baseline(U, n, out_pairs) -> Vector{Float64}

Analytical distinguishable-photon correlator baseline
`c^{dist}_{ij} = −Σ_{k=1}^{n} |U_{ik}|² |U_{jk}|²` indexed by the same
ordered pair list returned by `build_A_matrix_order_2`. This is what
must be subtracted from the empirical `c` before inverting `A b`.
"""
function distinguishable_correlator_baseline(U::AbstractMatrix, n::Integer,
                                              out_pairs::AbstractVector)
    c_dist = Vector{BSJ_FLOAT}(undef, length(out_pairs))
    @inbounds for (a, (i, j)) in enumerate(out_pairs)
        acc = 0.0
        for k in 1:n
            acc += abs2(U[i, k]) * abs2(U[j, k])
        end
        c_dist[a] = -acc
    end
    return c_dist
end

"""
    one_shot_bargmann_order_2(U_phys, samples, n)
        -> (b̂::Matrix{ComplexF64}, A, c, c_dist)

Appendix-A one-shot estimator: given the physics-convention
interferometer `U_phys` and `samples` from a single experiment with
`n` photons in the first `n` input modes, return the `n × n`
Hermitian matrix of estimated 2-Bargmann invariants,

    b̂[k, l] ≈ Tr(ρ_k ρ_l) = |G_{kl}|²     (k ≠ l),    b̂[k, k] = 1.

Internally: builds `A(U_phys)` (`eq:A(U)`), computes the empirical
connected correlators `c` and the analytical distinguishable baseline
`c^{dist}`, and solves `A · b = c − c^{dist}` by QR least-squares.
Returns also the raw `(A, c, c_dist)` for diagnostics.
"""
function one_shot_bargmann_order_2(U_phys::AbstractMatrix,
                                   samples::AbstractVector,
                                   n::Integer)
    m = size(U_phys, 1)
    A, out_pairs, in_pairs = build_A_matrix_order_2(U_phys, n)
    c = empirical_mode_correlators(samples, m, out_pairs)
    c_dist = distinguishable_correlator_baseline(U_phys, n, out_pairs)
    b_vec = A \ (c - c_dist)                # complex least-squares

    b_mat = Matrix{eltype(b_vec)}(I, n, n)  # diagonal = 1
    @inbounds for (idx, (k, l)) in enumerate(in_pairs)
        b_mat[k, l] = b_vec[idx]
    end
    # Symmetrise — the exact `b` is Hermitian (in fact real-symmetric, since
    # Tr(ρ_k ρ_l) = |G_{kl}|²); finite-N introduces small imaginary drift.
    b_mat = 0.5 * (b_mat + b_mat')
    return b_mat, A, c, c_dist
end

# ---------------------------------------------------------------------------
# Order 3 (Eq. 29 of Robbio 2025): triangle Bargmann invariants from
# 3-point correlators of a single experiment.
# ---------------------------------------------------------------------------

"""Sort three distinct integers into an ascending tuple, allocation-free."""
function _sorted_triple(i::Integer, j::Integer, k::Integer)
    a, b = minmax(i, j)
    a, c = minmax(a, k)
    b, c = minmax(b, c)
    return (a, b, c)
end

"""True iff the ordered triple `(i, j, k)` lies in the cyclic class of the
sorted triple `(min, mid, max)` of `{i, j, k}`. Cyclic class of the ascending
sorted `(r, s, t)` is `{(r,s,t), (s,t,r), (t,r,s)}` — exactly the *even*
permutations, so for distinct inputs membership is an inversion-parity test
(zero inversions = ascending; its two rotations have two inversions each)."""
_is_cyclic_to_sorted(i::Integer, j::Integer, k::Integer) =
    iseven((i > j) + (j > k) + (i > k))

"""
    predicted_3mode_moment(U_phys, G, a, b, c) -> ComplexF64

Eq. (29) of Robbio 2025 for `⟨n_a n_b n_c⟩`, distinct `(a, b, c)`.
Physics convention (`U[i, k]` = amplitude input `k` → output `i`). The
imaginary part is zero up to floating-point drift. Reference only — the
order-3 solver builds the Jacobian directly.
"""
function predicted_3mode_moment(U_phys::AbstractMatrix, G::AbstractMatrix,
                                a::Integer, b::Integer, c::Integer)
    n = size(G, 1)
    total = zero(promote_type(eltype(U_phys), eltype(G)))
    @inbounds for r in 1:n, s in 1:n, t in 1:n
        (r == s || s == t || r == t) && continue
        Ura = U_phys[a, r];  Usa = U_phys[a, s];  Uta = U_phys[a, t]
        Urb = U_phys[b, r];  Usb = U_phys[b, s];  Utb = U_phys[b, t]
        Urc = U_phys[c, r];  Usc = U_phys[c, s];  Utc = U_phys[c, t]
        total += abs2(Ura) * abs2(Usb) * abs2(Utc)
        total += abs2(Ura) * Usb * conj(Utb) * Utc * conj(Usc) * G[s, t] * G[t, s]
        total += Ura * conj(Usa) * Usb * conj(Urb) * abs2(Utc) * G[r, s] * G[s, r]
        total += Ura * conj(Uta) * abs2(Usb) * Utc * conj(Urc) * G[r, t] * G[t, r]
        total += Ura * conj(Usa) * Usb * conj(Utb) * Utc * conj(Urc) *
                 G[s, r] * G[r, t] * G[t, s]
        total += Ura * conj(Uta) * Usb * conj(Urb) * Utc * conj(Usc) *
                 G[s, t] * G[t, r] * G[r, s]
    end
    return total
end

"""
    build_A_matrix_order_3(U_phys, n) -> (A3_real, abc_triples, rst_triples)

Real Jacobian of `⟨n_a n_b n_c⟩` with respect to `(Re B_{rst}, Im B_{rst})`
for one canonical `B_{rst}` per unordered input triple `{r<s<t}`.

Returns `A3_real::Matrix{Float64}` of size `C(m,3) × 2·C(n,3)` and the
lex-ordered triple lists. Derivation: in Eq. (29), summing the six
orderings of `(r,s,t)` gives `α·B + conj(α)·conj(B) = 2 Re(α·B)` for
some `α(a,b,c; r,s,t)`; the two row entries per `B` are `(2 Re α, -2 Im α)`.
"""
function build_A_matrix_order_3(U_phys::AbstractMatrix, n::Integer)
    m = size(U_phys, 1)
    abc_triples = [(a, b, c) for a in 1:m for b in (a+1):m for c in (b+1):m]
    rst_triples = [(r, s, t) for r in 1:n for s in (r+1):n for t in (s+1):n]

    A3_real = zeros(BSJ_FLOAT, length(abc_triples), 2 * length(rst_triples))

    @inbounds for (row, (a, b, c)) in enumerate(abc_triples)
        for (idx, (r, s, t)) in enumerate(rst_triples)
            α = zero(eltype(U_phys))
            # Only orderings whose trace equals B_{rst} (cyclic to the sorted
            # canonical) add their coefficient to α; the others (`: zero` below)
            # carry conj(B) and are NOT discarded — they pair off as complex
            # conjugates of the kept terms and are restored by the
            # (2 Re α, -2 Im α) packing two lines down. See CLAUDE.md
            # § "Bargmann invariants — the conjugation trap".
            # Iterate over all 6 orderings (rp, sp, tp) of (r, s, t).
            for perm in ((r,s,t), (r,t,s), (s,r,t), (s,t,r), (t,r,s), (t,s,r))
                rp, sp, tp = perm
                # Term 5 coefficient and its trace argument ordering.
                coef5 = U_phys[a, rp] * conj(U_phys[a, sp]) *
                        U_phys[b, sp] * conj(U_phys[b, tp]) *
                        U_phys[c, tp] * conj(U_phys[c, rp])
                # Trace is Tr(ρ_{sp} ρ_{rp} ρ_{tp}); equals B if (sp, rp, tp)
                # is cyclic to the sorted canonical (r, s, t), else conj(B) —
                # only the kept (cyclic) terms add to α; the conj(B) terms are
                # restored by the (2 Re α, -2 Im α) packing below.
                _is_cyclic_to_sorted(sp, rp, tp) && (α += coef5)

                coef6 = U_phys[a, rp] * conj(U_phys[a, tp]) *
                        U_phys[b, sp] * conj(U_phys[b, rp]) *
                        U_phys[c, tp] * conj(U_phys[c, sp])
                # Trace Tr(ρ_{sp} ρ_{tp} ρ_{rp}); cyclic test on (sp, tp, rp).
                _is_cyclic_to_sorted(sp, tp, rp) && (α += coef6)
            end
            A3_real[row, 2*idx - 1] = 2 * real(α)     # coef of Re(B)
            A3_real[row, 2*idx]     = -2 * imag(α)    # coef of Im(B)
        end
    end
    return A3_real, abc_triples, rst_triples
end

"""
    three_mode_correlator_subtraction(U_phys, b2::AbstractMatrix,
                                      abc_triples, n) -> Vector{Float64}

For each output triple `(a, b, c)`, compute the Eq. (29) "known" part —
the distinguishable baseline and the 2-Bargmann contributions — given
the order-2 estimates `b2[r, s] ≈ Tr(ρ_r ρ_s) = |G_{rs}|²`. Returns a
vector of the same length as `abc_triples`. The 3-Bargmann contribution
to `⟨n_a n_b n_c⟩` is `empirical[abc] − subtraction[abc]`.
"""
function three_mode_correlator_subtraction(U_phys::AbstractMatrix,
                                           b2::AbstractMatrix,
                                           abc_triples::AbstractVector, n::Integer)
    y = zeros(BSJ_FLOAT, length(abc_triples))
    @inbounds for (row, (a, b, c)) in enumerate(abc_triples)
        acc = 0.0
        for r in 1:n, s in 1:n, t in 1:n
            (r == s || s == t || r == t) && continue
            Ura = U_phys[a, r];  Usa = U_phys[a, s];  Uta = U_phys[a, t]
            Urb = U_phys[b, r];  Usb = U_phys[b, s];  Utb = U_phys[b, t]
            Urc = U_phys[c, r];  Usc = U_phys[c, s];  Utc = U_phys[c, t]
            base = abs2(Ura) * abs2(Usb) * abs2(Utc)
            term_st = real(abs2(Ura) * Usb * conj(Utb) * Utc * conj(Usc)) * real(b2[s, t])
            term_rs = real(Ura * conj(Usa) * Usb * conj(Urb) * abs2(Utc)) * real(b2[r, s])
            term_rt = real(Ura * conj(Uta) * abs2(Usb) * Utc * conj(Urc)) * real(b2[r, t])
            acc += base + term_st + term_rs + term_rt
        end
        y[row] = acc
    end
    return y
end

"""
    empirical_3mode_correlators(samples, abc_triples) -> Vector{Float64}

Empirical `⟨n_a n_b n_c⟩` from mode-occupation samples, for each sorted
triple `(a < b < c)`.

Single pass over `samples`: only the `C(n, 3)` triples of occupied modes per
shot contribute, so we enumerate those and scatter into `y` through a 3-D
lookup table (`idx3[a,b,c] = row of (a,b,c) in abc_triples`). This is
`O(N·C(n,3))` versus the naive `O(C(m,3)·N)` of re-scanning `samples` once per
output triple. `abc_triples` is assumed lex-ordered (the output of
`build_A_matrix_order_3`), so its last entry fixes the mode count `m`.
"""
function empirical_3mode_correlators(samples::AbstractVector,
                                     abc_triples::AbstractVector)
    N = length(samples)
    ntri = length(abc_triples)
    y = zeros(BSJ_FLOAT, ntri)
    ntri == 0 && return y
    m = abc_triples[end][3]
    idx3 = zeros(Int, m, m, m)             # (a,b,c) → row; 0 = not requested
    @inbounds for (r, (a, b, c)) in enumerate(abc_triples)
        idx3[a, b, c] = r
    end
    occ = Vector{Int}(undef, m)            # reused occupied-mode buffer (ascending)
    @inbounds for s in samples
        k = 0
        for i in 1:m
            s[i] != 0 && (k += 1; occ[k] = i)
        end
        for x in 1:k, yy in (x+1):k, z in (yy+1):k
            a = occ[x]; b = occ[yy]; c = occ[z]   # ascending ⇒ matches sorted triple
            r = idx3[a, b, c]
            r != 0 && (y[r] += s[a] * s[b] * s[c])
        end
    end
    y ./= N
    return y
end

"""
    one_shot_bargmann_order_3(U_phys, samples, n, b2::AbstractMatrix)
        -> (B3::Dict, A3_real, residual::Float64)

Given samples from the SAME experiment used for order 2, and the order-2
estimates `b2[r, s] ≈ |G_{rs}|²`, recover the complex triangle Bargmann
invariants `B̂_{rst} = Tr(ρ_r ρ_s ρ_t)` for every sorted input triple
`r < s < t`.

Algorithm:
1. Build `A3_real`, `abc_triples`, `rst_triples` via `build_A_matrix_order_3`.
2. Compute the empirical `⟨n_a n_b n_c⟩` vector and the analytical
   subtraction (baseline + 2-Bargmann contributions).
3. Solve `A3_real · x = y_empirical − y_subtract` by QR least-squares.
   `x` alternates `(Re B_1, Im B_1, Re B_2, …)` over the sorted input
   triples.
4. Repack as a `Dict{NTuple{3, Int}, ComplexF64}` keyed by sorted triple.

Returns the Dict, the Jacobian matrix (for condition-number diagnostics),
and the Frobenius residual of the least-squares fit.
"""
function one_shot_bargmann_order_3(U_phys::AbstractMatrix,
                                    samples::AbstractVector,
                                    n::Integer,
                                    b2::AbstractMatrix)
    A3_real, abc_triples, rst_triples = build_A_matrix_order_3(U_phys, n)
    y_emp  = empirical_3mode_correlators(samples, abc_triples)
    y_sub  = three_mode_correlator_subtraction(U_phys, b2, abc_triples, n)
    x = A3_real \ (y_emp - y_sub)
    residual = norm(A3_real * x - (y_emp - y_sub))

    B3 = Dict{NTuple{3, Int}, BSJ_COMPLEX}()
    for (i, (r, s, t)) in enumerate(rst_triples)
        B3[(r, s, t)] = x[2*i - 1] + im * x[2*i]
    end
    return B3, A3_real, residual
end

"""
    bargmann_invariant_for_triple(B3::Dict, i, mid, j) -> ComplexF64

Return `Tr(ρ_i ρ_{mid} ρ_j)` given a dictionary keyed by sorted triples
`(r < s < t)` mapping to `B_{rst} = Tr(ρ_r ρ_s ρ_t)`. Uses the cyclic
symmetry of the trace: if `(i, mid, j)` is in the cyclic class of the
sorted canonical, return `B`; otherwise return `conj(B)`.
"""
function bargmann_invariant_for_triple(B3::Dict{NTuple{3, Int}, BSJ_COMPLEX},
                                       i::Integer, mid::Integer, j::Integer)
    sorted = _sorted_triple(i, mid, j)
    @assert haskey(B3, sorted) "triple $sorted missing from B3"
    return _is_cyclic_to_sorted(i, mid, j) ? B3[sorted] : conj(B3[sorted])
end
