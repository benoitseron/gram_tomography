"""
Gram matrix, Bargmann invariants, tree-gauge assembly.

State vectors are columns of `Ψ ∈ ℂ^{d × n}`; `G = Ψ'*Ψ` gives
`G_{ij} = ⟨ψ_i|ψ_j⟩` with `G_{ji} = conj(G_{ij})`. Ideal Bargmann
invariant along `π = (u₁,…,u_k)`:

    B_π = Tr(ρ_{u₁}⋯ρ_{u_k}) = G_{u₁,u₂} G_{u₂,u₃} ⋯ G_{u_{k-1},u_k} G_{u_k,u₁}

(every factor read off `G` in order, no explicit `conj`; for `k = 2`
reduces to `|G_{ij}|²`).

Tree gauge: `G_{i,j} ∈ ℝ_{≥0}` on tree edges. Phase of `B_{π(i,j)}` is
then **−arg(G_{ij})** (closing factor `G_{j,i} = conj(G_{i,j})`), so
`assemble_gram_from_bargmann` conjugates `B̂_π` before extracting the
phase — see its docstring. Full long-form convention discussion in
`CLAUDE.md` § "Conjugate & Adjoint Conventions".
"""

using LinearAlgebra
using Random
using Graphs: SimpleGraph, add_edge!, dijkstra_shortest_paths, enumerate_paths

"""
    random_gram_matrix(n::Integer, d::Integer;
                       T::Type{<:Complex}=ComplexF64, rng=Random.default_rng())

Return `(G, Ψ)` where `Ψ` is a `d × n` matrix whose columns are uniform random
unit vectors in ℂ^d and `G = Ψ' * Ψ` is their Gram matrix. The element type `T`
defaults to `ComplexF64`; pass e.g. `T=Complex{BigFloat}` to exercise the
high-precision path of the pure-math modules.
"""
function random_gram_matrix(n::Integer, d::Integer;
                            T::Type{<:Complex}=ComplexF64, rng=Random.default_rng())
    Ψ = randn(rng, T, d, n)
    for col in eachcol(Ψ)
        normalize!(col)
    end
    G = Ψ' * Ψ
    return G, Ψ
end

"""
    bargmann_invariant(G::AbstractMatrix, path::AbstractVector{<:Integer})

Ideal Bargmann invariant along `path = (u₁,…,u_k)`:

    B_π = G_{u_1,u_2} · G_{u_2,u_3} ⋯ G_{u_{k-1},u_k} · G_{u_k,u_1}

Reduces to `G_{ij} G_{ji} = |G_{ij}|²` when `path = [i, j]`
(`B_{i₁…i_m} := Tr(ψ_{i₁}⋯ψ_{i_m})`, `notes/gtomo_new/Sections/gram-estimation.tex`;
Tr(ρ_{u_1}⋯ρ_{u_k}) for rank-one ρ_l).
"""
function bargmann_invariant(G::AbstractMatrix, path::AbstractVector{<:Integer})
    @assert length(path) ≥ 2
    val = one(eltype(G))
    for l in 1:(length(path)-1)
        val *= G[path[l], path[l+1]]
    end
    val *= G[path[end], path[1]]
    return val
end

"""
    tree_path(T::AbstractMatrix{Bool}, i::Integer, j::Integer) -> Vector{Int}

Return the unique path (as a vertex sequence `[i, …, j]`) between `i` and `j`
in the spanning tree whose adjacency is `T` (symmetric Boolean matrix).

On a tree the unique path equals the (unweighted) shortest path, so this
delegates path-finding to `Graphs.jl` (`dijkstra_shortest_paths` +
`enumerate_paths`). The `SimpleGraph` is built inline rather than via
`spanning_tree.jl`'s `_graph` helper because this file is `include`d first.
"""
function tree_path(T::AbstractMatrix{Bool}, i::Integer, j::Integer)
    i == j && return [Int(i)]                        # trivial self-path
    n = size(T, 1)
    g = SimpleGraph(n)
    @inbounds for a in 1:n, b in (a+1):n
        T[a, b] && add_edge!(g, a, b)
    end
    path = enumerate_paths(dijkstra_shortest_paths(g, i), j)
    isempty(path) && error("vertices $i and $j are not connected in T")
    return path
end

"""
    regauge_to_tree(G::AbstractMatrix, T::AbstractMatrix{Bool}; root::Integer=1)

Return the Gram matrix `G'` obtained by absorbing per-state phases
`ψ_i ↦ e^{-iθ_i} ψ_i` so that every tree edge `(u, v) ∈ T` has
`G'[u, v] = G'[v, u]^* ∈ ℝ_{≥0}` — i.e. the tree-edge gauge used by
`assemble_gram_from_bargmann`. The BFS builds `θ` starting from `root`
(default vertex 1) so that `θ_v = θ_parent + arg(G[parent, v])`.

In matrix form, `G' = D · G · D†` with `D = diag(e^{iθ})`. This is a
pure gauge transformation; all Bargmann invariants (closed-cycle traces)
are unchanged, and the equivalence class of pure-state Gram matrices is
preserved.

Use this in tests to compare the reconstructed `Ĝ` — which is tied to
its own MDST — against the true `G` after matching gauges.
"""
function regauge_to_tree(G::AbstractMatrix, T::AbstractMatrix{Bool}; root::Integer=1)
    n = size(G, 1)
    θ = zeros(real(float(eltype(G))), n)
    parent = fill(0, n); parent[root] = root
    queue = [root]
    while !isempty(queue)
        v = popfirst!(queue)
        for u in 1:n
            if T[v, u] && parent[u] == 0
                parent[u] = v
                θ[u] = θ[v] + angle(G[v, u])
                push!(queue, u)
            end
        end
    end
    D = Diagonal(exp.(im .* θ))
    return D * G * D'
end

"""
    assemble_gram_from_bargmann(B̂_edge, B̂_path, edges, T, n) -> Matrix{ComplexF64}

Assemble `Ĝ` from Bargmann-invariant estimates. Eqs. `eq:gram1`
(tree) and `eq:gram2` (non-tree) of `notes/gtomo_new/Sections/gram-estimation.tex`
(= Eqs. (32)/(34) and (33)/(35) of `Gram_Tomography.pdf`):

- tree edge:      `Ĝ_{ij} = √B̂_{ij}`  (non-negative real).
- non-tree edge:  `Ĝ_{ij} = (conj(B̂_π) / |B̂_π|) · √B̂_{ij}`, where
  `π = π(i,j)` is the unique tree path; the `conj` flips the phase
  back to `+arg(G_{ij})` from the `-arg(G_{ij})` induced by the closing
  factor of the trace.

Inputs: `B̂_edge`, `B̂_path :: Dict{(i<j) => ComplexF64}`, edge list,
tree adjacency, vertex count. Result is Hermitian with unit diagonal,
not necessarily PSD — post-compose with `project_gram` for a valid
Gram matrix.
"""
function assemble_gram_from_bargmann(B̂_edge::AbstractDict, B̂_path::AbstractDict,
                                     edges::AbstractVector, T::AbstractMatrix{Bool}, n::Integer)
    # Output holds complex phases; element type follows the inputs (a Gram
    # matrix is complex even when the edge magnitudes arrive real).
    Tc = complex(float(promote_type(valtype(B̂_edge), valtype(B̂_path))))
    Ĝ = Matrix{Tc}(I, n, n)
    for (i, j) in edges
        bij = B̂_edge[(i, j)]
        magn = sqrt(max(real(bij), zero(real(Tc))))
        if T[i, j]
            val = Tc(magn)
        else
            bπ = B̂_path[(i, j)]
            # Use conj(bπ)/|bπ| so the phase factor equals +arg(G_{ij}) under
            # our Tr(ρ_{u_1}⋯ρ_{u_k}) convention; see long-form docstring above.
            phase = iszero(bπ) ? one(Tc) : conj(bπ) / abs(bπ)
            val = phase * magn
        end
        Ĝ[i, j] = val
        Ĝ[j, i] = conj(val)
    end
    return Ĝ
end
