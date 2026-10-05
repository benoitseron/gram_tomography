"""
Supplier-agnostic Gram-matrix tomography protocol.

Implements the boxed "Gram matrix tomography protocol" of
`notes/gtomo_new/Sections/gram-estimation.tex` (steps 2–6):

2. δ-threshold: remove edges with `|B̂_{ij}| < δ`         → `protocol_graph`
3. optimal spanning tree (MDST, `sec:tree`)               → `protocol_tree`
4. list the tree-path invariants `B_{π(i,j)}` to measure  → `required_paths`
5. assemble Ĝ from `eq:gram1` / `eq:gram2`                ┐
6. project onto the correlation matrices (Higham)         ┘ `reconstruct_gram`

The protocol consumes Bargmann-invariant *estimates* and is agnostic to how
they were measured (step 1 / step 4 supplier: correlator method of
`Accessing_relation_information.tex`, Fourier method, synthetic noise, …):

- `B2[i, j] ≈ B_{ij} = |G_{ij}|²`   — order-2 invariants, all pairs;
- `Bpath[(i, j)] ≈ B_{π(i,j)}`      — full complex invariant along the unique
  tree path `π(i,j)`, for every surviving non-tree edge `i < j`.

Conventions are those of `gram_matrix.jl` (no explicit `conj` in Bargmann
products; the tree-gauge phase sign is handled inside
`assemble_gram_from_bargmann`); see `CLAUDE.md` § "Conjugate & Adjoint
Conventions".
"""

using LinearAlgebra
using Graphs: is_connected

"""
    protocol_graph(B2::AbstractMatrix, δ::Real) -> (Γ::BitMatrix, n_removed::Int)

Protocol step 2 of `gram-estimation.tex`: "Fix a threshold δ and remove edges
with |B̂_{ij}| < δ". Returns the surviving graph

    Γ[i, j] = (i ≠ j) && (|B2[i, j]| ≥ δ)        (diagonal false)

and `n_removed = n(δ)`, the number of unordered pairs `i < j` removed — the
count entering the truncation term `n(δ)·δ` of Theorem 1 (eq. 40). Taking
`δ ≤ δ_min` (`eq:deltamin`) keeps every true edge and sets `n(δ) = 0`.
"""
function protocol_graph(B2::AbstractMatrix, δ::Real)
    n = size(B2, 1)
    @assert size(B2, 2) == n "B2 must be square"
    Γ = falses(n, n)
    n_removed = 0
    for i in 1:n, j in (i+1):n
        if abs(B2[i, j]) ≥ δ
            Γ[i, j] = true
            Γ[j, i] = true
        else
            n_removed += 1
        end
    end
    return Γ, n_removed
end

"""
    protocol_tree(Γ::AbstractMatrix{Bool}, W::AbstractMatrix) -> Matrix{Bool}

Protocol step 3: the optimal spanning tree of the surviving graph `Γ` under
weights `W[i, j] = |B̂_{ij}|`, i.e. the MDST on `γ(u, v) = -log W[u, v]`
(`sec:tree` of `gram-estimation.tex`). Errors if `Γ` is disconnected: the
manuscript assumes the post-cutoff graph is connected ("The remaining pairs
(i,j) form a graph Γ, which we again assume to be connected") — lower δ, or
run the protocol on each connected component separately.
"""
function protocol_tree(Γ::AbstractMatrix{Bool}, W::AbstractMatrix)
    # Connectivity check via Graphs.jl (`_graph` from spanning_tree.jl, the same
    # helper min_diameter_spanning_tree uses below).
    is_connected(_graph(Γ)) ||
        error("graph Γ is disconnected after the δ cutoff; the " *
              "manuscript (gram-estimation.tex) assumes Γ connected — " *
              "lower δ or run the protocol per connected component")
    return min_diameter_spanning_tree_bargmann(W, Γ)
end

"""
    required_paths(Γ::AbstractMatrix{Bool}, T::AbstractMatrix{Bool})
        -> Vector{Tuple{Tuple{Int,Int}, Vector{Int}}}

Protocol step 4 (the measurement list): for each surviving non-tree edge
`(i, j)`, `i < j`, with `Γ[i, j] && !T[i, j]`, return `((i, j), π(i, j))`
where `π(i, j)` is the unique tree path joining `i` and `j`. These are the
Bargmann invariants `B_{π(i,j)}` the supplier must provide to fix the phases
via `eq:estimator-phase` / `eq:gram2` (`gram-estimation.tex`).
"""
function required_paths(Γ::AbstractMatrix{Bool}, T::AbstractMatrix{Bool})
    n = size(Γ, 1)
    reqs = Tuple{Tuple{Int, Int}, Vector{Int}}[]
    for i in 1:n, j in (i+1):n
        if Γ[i, j] && !T[i, j]
            push!(reqs, ((i, j), tree_path(T, i, j)))
        end
    end
    return reqs
end

"""
    plan_required_cycles(B2::AbstractMatrix; δ::Real=0.0)
        -> (; Γ, T, n_removed, cycles)

**Phase 1** of the two-phase protocol (steps 2–4 of `gram-estimation.tex`),
packaged as a standalone deliverable. Given order-2 estimates
`B2[i,j] ≈ |G_{ij}|²` for every pair, it applies the δ-cutoff
(`protocol_graph`), builds the MDST (`protocol_tree`), and returns the
explicit list of cycle invariants that still have to be measured:

- `Γ`, `T`     — surviving graph and its optimal spanning tree;
- `n_removed`  — `n(δ)`, edges dropped by the cutoff;
- `cycles`     — `Vector` of `(; edge=(i,j), path, order=length(path))`, one
                 per surviving non-tree edge: the Bargmann invariant
                 `B_{π(i,j)} = Tr(ρ_i ρ_… ρ_j)` of the given `order` that the
                 caller must supply.

This makes the protocol's inherent two-experiment structure first-class. The
pairwise `B2` come from one measurement (step 1, ANY estimator — one-shot,
Fourier, …); `plan_required_cycles` then says *exactly which* higher-order
cycles to go and measure (possibly in a separate experimental run, with
whatever estimator suits each order — the `order` field is there to route
each cycle, e.g. one-shot for order 3, Fourier for order > 3); finally
`reconstruct_gram(B2, Bpath, δ)` (steps 5–6) assembles `Ĝ`.
`plan_required_cycles` and `reconstruct_gram` recompute `Γ`, `T` identically
from `(B2, δ)` (deterministic), so the `cycles` plan matches exactly the
`Bpath` keys `reconstruct_gram` expects.

Generically (random states) every cycle has `order ≤ 3`; engineered
partially-distinguishable states can force higher orders (see the manuscript
discussion) — which is the only situation where the Fourier method is needed.
"""
function plan_required_cycles(B2::AbstractMatrix; δ::Real=0.0)
    Γ, n_removed = protocol_graph(B2, δ)                # step 2
    W = abs.(B2)
    T = protocol_tree(Γ, W)                             # step 3
    cycles = [(; edge=(i, j), path=path, order=length(path))   # step 4
              for ((i, j), path) in required_paths(Γ, T)]
    return (; Γ, T, n_removed, cycles)
end

"""
    reconstruct_gram(B2::AbstractMatrix, Bpath::AbstractDict, δ::Real)
        -> (; Ĝ, T, Γ, n_removed)

Protocol steps 2–6 of the boxed "Gram matrix tomography protocol"
(`gram-estimation.tex`), given the Bargmann-invariant estimates:

- `B2[i, j] ≈ B_{ij} = |G_{ij}|²` for every pair (step 1's output);
- `Bpath :: Dict{Tuple{Int,Int}, ComplexF64}`, keyed by `(i, j)` with
  `i < j` for each surviving non-tree edge, holding the full complex
  `B_{π(i,j)} = Tr(ρ_i ρ_… ρ_j)` along the tree path (step 4's output —
  see `required_paths`).

Pipeline: δ-cutoff (`protocol_graph`) → MDST (`protocol_tree`) → assembly of
`Ĝ` from `eq:gram1` (tree edges, `Ĝ_{ij} = √B̂_{ij}`) and `eq:gram2`
(non-tree, phase × magnitude) via `assemble_gram_from_bargmann` → Higham
projection onto the correlation matrices (`project_gram`; the contraction
`‖p(Ĝ) − G‖₂ ≤ ‖Ĝ − G‖₂` holds since `G` is itself a correlation matrix).

Removed edges and non-edges stay 0 in the assembled matrix (the assembly
starts from the identity and fills only surviving edges), implementing
"set Ĝ_{ij} = 0 for all (i, j) with |B̂_{ij}| < δ".
"""
function reconstruct_gram(B2::AbstractMatrix, Bpath::AbstractDict, δ::Real)
    n = size(B2, 1)
    Γ, n_removed = protocol_graph(B2, δ)               # step 2
    W = abs.(B2)
    T = protocol_tree(Γ, W)                            # step 3
    for ((i, j), path) in required_paths(Γ, T)         # step 4 (inputs check)
        haskey(Bpath, (i, j)) ||
            error("Bpath is missing B_π for non-tree edge ($i, $j) " *
                  "along tree path $(path)")
    end
    edges = [(i, j) for i in 1:n for j in (i+1):n if Γ[i, j]]
    # Magnitudes² are real ≥ 0; the √ of eq:gram1/eq:gram2 is taken inside
    # `assemble_gram_from_bargmann`, which lifts these reals to the complex Ĝ.
    B̂_edge = Dict((i, j) => abs(B2[i, j]) for (i, j) in edges)
    Ĝ_raw = assemble_gram_from_bargmann(B̂_edge, Bpath, edges, T, n)  # step 5
    Ĝ, _, _ = project_gram(Ĝ_raw)                                    # step 6
    return (; Ĝ = Ĝ, T = T, Γ = Γ, n_removed = n_removed)
end
