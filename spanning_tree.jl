"""
Spanning-tree algorithms for the tomography protocol.

Implements the Minimum Diameter Spanning Tree (MDST) reduction from section
III C of `notes/Gram_Tomography.pdf` (section C):

- weights `w(i,j) = B_{ij} ∈ [0, 1]` (edge Bargmann invariants),
- `γ(u,v) = -log w(u,v) ≥ 0`,
- minimise `max_{π ∈ T} Σ γ(u_l, u_{l+1})`.

All the *textbook* graph primitives are delegated to `Graphs.jl` (the standard
Julia graph library): Kruskal's MST, Floyd–Warshall all-pairs shortest paths,
Dijkstra shortest-path trees, and the weighted tree diameter (two Dijkstras).
The Graphs.jl routines are generic over the weight `eltype`, so `BigFloat`
weight matrices flow through unchanged (see `core_precision_genericity`).

The only hand-written algorithm left is the MDST itself: no Julia package
implements Minimum Diameter Spanning Tree, so `min_diameter_spanning_tree`
keeps the absolute-1-center reduction (the manuscript's `[mdst]`, Hassin–Tamir)
on top of Graphs.jl's Floyd–Warshall + Dijkstra.

Both tree builders return just the tree `T` (a Boolean adjacency matrix). The
quantity entering the Thm 1 bound is computed separately on demand:

- `M_T_thm1(T, W)` — `M(T) = (min_{π∈T} ∏ √|B_{u_l,u_{l+1}}|)^{-1}` (eq. 38,
  sqrt-inside-the-product),
- `m_T(T)` — the hop-count diameter + 1 (eq. 41).
"""

using Graphs: SimpleGraph, add_edge!, add_vertex!, src, dst,
              is_connected, kruskal_mst, dijkstra_shortest_paths,
              floyd_warshall_shortest_paths
using LinearAlgebra

"""
    _graph(A::AbstractMatrix{Bool}) -> SimpleGraph

Build an (unweighted) `Graphs.SimpleGraph` from a symmetric Boolean adjacency
matrix. Edge weights are always supplied separately as a distance matrix to the
Graphs.jl routines, so the graph only carries the topology.
"""
function _graph(A::AbstractMatrix{Bool})
    n = size(A, 1)
    g = SimpleGraph(n)
    @inbounds for i in 1:n, j in (i+1):n
        A[i, j] && add_edge!(g, i, j)
    end
    return g
end

"""
    _tree_from_parents(parents, n) -> BitMatrix

Turn a Graphs.jl Dijkstra `parents` vector into a symmetric Boolean adjacency
matrix over vertices `1:n`. The source has `parents[src] == 0`; any parent
index `> n` (the virtual 1-center vertex, see `min_diameter_spanning_tree`) is
dropped here and stitched back by the caller.
"""
function _tree_from_parents(parents, n)
    T = falses(n, n)
    @inbounds for v in 1:n
        p = parents[v]
        if p != 0 && p <= n
            T[v, p] = T[p, v] = true
        end
    end
    return T
end

"""
    tree_diameter(T::AbstractMatrix{Bool}, γ::AbstractMatrix{<:Real})

Return the longest-path (diameter) weight of tree `T` under edge weights `γ`,
and the vertex pair realising it. Standard two-sweep trick: a shortest-path
sweep from any vertex reaches one diameter endpoint `u`; a second sweep from
`u` reaches the other. Both sweeps are Graphs.jl `dijkstra_shortest_paths`
(weights `γ ≥ 0` here, and on a tree the shortest path is the unique path).
"""
function tree_diameter(T::AbstractMatrix{Bool}, γ::AbstractMatrix{<:Real})
    g = _graph(T)
    d1 = dijkstra_shortest_paths(g, 1, γ).dists
    u = argmax(d1)
    d2 = dijkstra_shortest_paths(g, u, γ).dists
    v = argmax(d2)
    return d2[v], (u, v)
end

"""
    max_weight_spanning_tree(W::AbstractMatrix, Γ::AbstractMatrix{Bool}) -> BitMatrix

Return a maximum-weight spanning tree `T` of the graph with adjacency `Γ` and
weights `W[i,j] = B_{ij}`, via Graphs.jl `kruskal_mst(g, W; minimize=false)`.
Call `M_T_thm1(T, W)` for the eq.-38 cost.

This is NOT an MDST in general; use `min_diameter_spanning_tree` for the
rigorous optimum.
"""
function max_weight_spanning_tree(W::AbstractMatrix, Γ::AbstractMatrix{Bool})
    n = size(W, 1)
    g = _graph(Γ)
    is_connected(g) || error("graph is not connected")
    T = falses(n, n)
    for e in kruskal_mst(g, W; minimize = false)
        T[src(e), dst(e)] = T[dst(e), src(e)] = true
    end
    return T
end

"""
    min_diameter_spanning_tree(γ::AbstractMatrix{<:Real}, Γ::AbstractMatrix{Bool}) -> BitMatrix

EXACT minimum-diameter spanning tree of the weighted graph with adjacency `Γ`
and non-negative edge weights `γ`, returned as a Boolean adjacency matrix `T`.
The "diameter" is the longest tree path measured as the SUM of its edge
weights, so this minimises `max_{π ∈ T} Σ_l γ(u_l, u_{l+1})`.

This is the generic, weight-agnostic routine. For the tomography protocol the
weights are `γ(i,j) = -log|B_{ij}|`; call the
`min_diameter_spanning_tree_bargmann(W, Γ)` wrapper, which builds `γ` from the
Bargmann magnitudes `W` and then defers here (`sec:tree` of
`notes/gtomo_new/Sections/gram-estimation.tex`, eq. 43–46 of
`Gram_Tomography.pdf`).

Input guards (issue #14): `Γ` must be connected — the protocol assumes a
connected graph throughout, so a disconnected `Γ` is an error rather than a
silent partial tree. Every present edge must carry a finite, non-negative
weight; a non-finite/negative weight on an actual edge is an input clash (e.g.
a zero Bargmann invariant `γ = -log 0 = ∞` that the δ cutoff should have
removed) and also errors.

Algorithm — absolute-1-center reduction (Hassin–Tamir, the `[mdst]` reference
of the manuscript): the MDST is a shortest-path tree rooted at the *absolute
1-center* of the graph — the point `p` (possibly **inside** an edge) minimising
the eccentricity `ecc(p) = max_z d(p, z)` — and its diameter equals `2·ecc(p)`.
Proof sketch of optimality: any SPT from `p` has diameter `≤ 2·ecc(p)`; and any
spanning tree of diameter `D` contains a point of graph-eccentricity `≤ D/2`
(the tree's absolute center), so `min diameter = 2·min eccentricity`.

Implementation: all-pairs shortest paths via Graphs.jl Floyd–Warshall, then for
every edge `(u, v)` minimise the piecewise-linear eccentricity
`f(t) = max_z min(d(u,z) + t, d(v,z) + γ_uv − t)` over `t ∈ [0, γ_uv]`; the
minimum is attained at an endpoint or at a crossing of a rising branch
`d(u,z₁) + t` with a falling branch `d(v,z₂) + γ_uv − t`, all of which are
enumerated. Finally grow a Graphs.jl Dijkstra SPT from the optimal point (a
virtual vertex inside the optimal edge, collapsed back onto the edge after).

Exactness on small graphs is locked by the brute-force Prüfer-enumeration
testset in `tests.jl` ("MDST is optimal (brute force, n ≤ 7)"). The previous
vertex-rooted-only sweep (a 2-approximation) was caught suboptimal there.

Complexity: O(n³) for Floyd–Warshall + O(m·n²) breakpoints with O(n)
evaluation each — fine for the protocol-sized graphs used here (the
manuscript's `[mdst]` reference gives the asymptotically better
O(mn + n² log n) variant).
"""
function min_diameter_spanning_tree(γ::AbstractMatrix{<:Real}, Γ::AbstractMatrix{Bool})
    n = size(γ, 1)
    size(γ, 2) == n || error("γ must be square")
    size(Γ) == (n, n) || error("Γ must have the same size as γ")
    R = float(eltype(γ))

    g = _graph(Γ)
    is_connected(g) ||
        error("graph Γ is disconnected; the MDST is only defined on a " *
              "connected graph (run the protocol per connected component)")

    @inbounds for u in 1:n, v in (u+1):n
        Γ[u, v] || continue
        w = γ[u, v]
        (isfinite(w) && w ≥ 0) ||
            error("edge ($u,$v) is present in Γ but carries a non-finite/" *
                  "negative weight γ=$w (input clash; see issue #14)")
    end

    d = floyd_warshall_shortest_paths(g, γ).dists   # all-pairs shortest paths

    # Absolute 1-center: minimise ecc over every point of every edge.
    best_ecc = R(Inf)
    best_u, best_v, best_t = 0, 0, zero(R)
    for u in 1:n, v in (u+1):n
        Γ[u, v] || continue
        w = γ[u, v]
        cands = R[zero(R), w]
        for z1 in 1:n, z2 in 1:n
            t = (d[v, z2] + w - d[u, z1]) / 2
            (isfinite(t) && zero(R) < t < w) && push!(cands, t)
        end
        for t in cands
            ecc = zero(R)
            for z in 1:n
                e = min(d[u, z] + t, d[v, z] + w - t)
                e > ecc && (ecc = e)
            end
            if ecc < best_ecc
                best_ecc = ecc
                best_u, best_v, best_t = u, v, t
            end
        end
    end
    best_u == 0 && return falses(n, n)        # no finite edge (degenerate input)

    # Shortest-path tree from the absolute 1-center.
    u, v, t = best_u, best_v, best_t
    w = γ[u, v]
    if t <= 0 || t >= w
        src = t <= 0 ? u : v
        return _tree_from_parents(dijkstra_shortest_paths(g, src, γ).parents, n)
    end
    # Center strictly inside edge (u, v): Dijkstra from a virtual vertex n+1
    # splitting the edge into pieces t and w−t, then collapse it.
    ga = copy(g)
    add_vertex!(ga)
    add_edge!(ga, u, n + 1)
    add_edge!(ga, v, n + 1)
    γa = fill(R(Inf), n + 1, n + 1)
    γa[1:n, 1:n] .= γ
    γa[n+1, u] = γa[u, n+1] = t
    γa[n+1, v] = γa[v, n+1] = w - t
    parents = dijkstra_shortest_paths(ga, n + 1, γa).parents
    T = _tree_from_parents(parents, n)
    if parents[u] == n + 1 && parents[v] == n + 1
        T[u, v] = T[v, u] = true               # both halves used → real edge
    end
    return T
end

"""
    min_diameter_spanning_tree_bargmann(W::AbstractMatrix, Γ::AbstractMatrix{Bool}) -> BitMatrix

Protocol wrapper around `min_diameter_spanning_tree`: builds the MDST distance
weights `γ(i,j) = -log|W[i,j]|` from the Bargmann magnitudes `W[i,j] = |B_{ij}|`
on the edges of `Γ`, then defers to the generic routine. This solves the MDST
of `sec:tree` of `gram-estimation.tex` (eq. 43–46 of `Gram_Tomography.pdf`).
Call `M_T_thm1(T, W)` for the eq.-38 cost.

A present edge with `|W[i,j]| == 0` is an input clash (issue #14): the zero
Bargmann invariant would map to `γ = -log 0 = ∞`, which is indistinguishable
from an absent edge. The δ cutoff (`eq:deltamin`) is supposed to remove such
edges before tree construction, so this errors rather than silently dropping
the edge.
"""
function min_diameter_spanning_tree_bargmann(W::AbstractMatrix, Γ::AbstractMatrix{Bool})
    n = size(W, 1)
    R = float(real(eltype(W)))
    γ = fill(R(Inf), n, n)            # absent edges stay at +∞ (never read)
    @inbounds for u in 1:n, v in (u+1):n
        Γ[u, v] || continue
        w = abs(W[u, v])
        w > 0 ||
            error("edge ($u,$v) is present in Γ but has zero Bargmann weight " *
                  "(|W|=0); the δ cutoff (eq:deltamin) should remove such " *
                  "edges before tree construction (issue #14)")
        γ[u, v] = γ[v, u] = -log(w)
    end
    return min_diameter_spanning_tree(γ, Γ)
end

"""
    M_T_thm1(T::AbstractMatrix{Bool}, W::AbstractMatrix) -> Real

Thm 1 `M(T)` (eq. 38 of `Gram_Tomography.pdf`):

    M(T)  =  ( min_{π ∈ T} ∏_l √|W[u_l, u_{l+1}]| )^{-1}

Computed directly as `exp(diameter)` under the half-log weights
`γ_half = -½ log|W|` — i.e. the longest tree path under `γ_half`. Equivalently
the square root of `(min_π ∏ |W|)^{-1}`, the no-sqrt cost; the sqrt-inside-the-
product is what absorbs the bound-audit's missing `M/δ` factor.
"""
function M_T_thm1(T::AbstractMatrix{Bool}, W::AbstractMatrix)
    R = float(real(eltype(W)))
    γ_half = similar(W, R)
    @inbounds for I in eachindex(W)
        w = W[I]
        γ_half[I] = w > 0 ? -log(abs(w)) / 2 : R(Inf)
    end
    diam, _ = tree_diameter(T, γ_half)
    return exp(diam)
end

"""
    m_T(T::AbstractMatrix{Bool}) -> Int

Hop-count diameter of tree `T` plus one (eq. 41 of `Gram_Tomography.pdf`):
the length of the longest path in the tree (counted in edges) plus one. Equals
the maximal *order* of Bargmann invariant the protocol must measure (a path of
`k` edges visits `k+1` vertices, so we need a Bargmann invariant of order
`k+1 = m(T)`).
"""
function m_T(T::AbstractMatrix{Bool})
    n = size(T, 1)
    γ_unit = ones(Int, n, n)
    diam, _ = tree_diameter(T, γ_unit)
    return Int(round(diam)) + 1
end
