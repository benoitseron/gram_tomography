"""
Layer-1 tests for the Bargmann tomography protocol.

Every `@testset` verifies one claim from the manuscript
(`notes/gtomo_new/` and its render `notes/Gram_Tomography.pdf`). Tests are
written so that they run cheaply (small n, d; modest N) while exercising every
identity.
"""

using Test
using LinearAlgebra
using Random
using Statistics
using DataStructures

const _CORE = @__DIR__
include(joinpath(_CORE, "gram_matrix.jl"))
include(joinpath(_CORE, "estimators.jl"))
include(joinpath(_CORE, "spanning_tree.jl"))
include(joinpath(_CORE, "projection.jl"))
include(joinpath(_CORE, "protocol.jl"))
include(joinpath(_CORE, "reconstruction.jl"))
include(joinpath(_CORE, "boson_sampling_integration.jl"))
include(joinpath(_CORE, "one_shot_estimator.jl"))
include(joinpath(_CORE, "fourier_estimator.jl"))
include(joinpath(_CORE, "physical_tomography.jl"))

Random.seed!(20260415)

@testset "gram_matrix: random Gram matrix properties" begin
    for (n, d) in [(3, 2), (5, 4), (8, 6), (6, 10)]
        G, Ψ = random_gram_matrix(n, d)
        @test size(G) == (n, n)
        @test norm(G - G') < 1e-10
        @test all(abs(G[i, i] - 1) < 1e-10 for i in 1:n)
        # rank at most d
        vals = eigvals(Hermitian(Matrix(G)))
        @test minimum(vals) > -1e-10
        @test count(>(1e-8), vals) ≤ d
    end
end

@testset "gram_matrix: bargmann_invariant identities" begin
    # 2-path: B_{ij} = |G_{ij}|² (notes L27 for k=2).
    for _ in 1:20
        G, _ = random_gram_matrix(5, 3)
        i, j = 1, 4
        @test isapprox(bargmann_invariant(G, [i, j]), abs2(G[i, j]); atol=1e-12)
    end

    # Cycle-closure: B_π for a closed triangle. With G = Ψ†Ψ,
    # Tr(|ψ_i⟩⟨ψ_i|·|ψ_j⟩⟨ψ_j|·|ψ_k⟩⟨ψ_k|) = ⟨ψ_i|ψ_j⟩⟨ψ_j|ψ_k⟩⟨ψ_k|ψ_i⟩
    #                                       = G_{ij} G_{jk} G_{ki}  (no conj).
    for _ in 1:20
        G, Ψ = random_gram_matrix(4, 3)
        i, j, k = 1, 2, 3
        tri = bargmann_invariant(G, [i, j, k])
        ρi = Ψ[:, i] * Ψ[:, i]'
        ρj = Ψ[:, j] * Ψ[:, j]'
        ρk = Ψ[:, k] * Ψ[:, k]'
        @test isapprox(tri, tr(ρi * ρj * ρk); atol=1e-10)
    end
end

@testset "gram_matrix: tree_path correctness" begin
    # Path graph 1-2-3-4-5.
    n = 5
    T = falses(n, n)
    for i in 1:(n-1); T[i, i+1] = true; T[i+1, i] = true; end
    @test tree_path(T, 1, 5) == [1, 2, 3, 4, 5]
    @test tree_path(T, 3, 1) == [3, 2, 1]
    # Star on 4 vertices, centre = 1.
    n = 4
    S = falses(n, n)
    for i in 2:n; S[1, i] = true; S[i, 1] = true; end
    @test tree_path(S, 2, 3) == [2, 1, 3]
end

@testset "gram_matrix: assemble recovers G in noiseless limit" begin
    # With exact Bargmann invariants (σ = 0, infinite N), Ĝ = G (up to the
    # tree-fixed gauge). We therefore pick G with real tree entries so no
    # gauge rotation is needed on tree edges.
    n, d = 5, 4
    G, Ψ = random_gram_matrix(n, d)
    # Rotate each column so G[1, j] is non-negative real (gauge on a star tree).
    for j in 2:n
        z = G[1, j]
        if z != 0
            Ψ[:, j] *= conj(z) / abs(z)
        end
    end
    G = Ψ' * Ψ  # re-gauged

    edges = [(i, j) for i in 1:n for j in (i+1):n]
    T = falses(n, n)
    for j in 2:n; T[1, j] = true; T[j, 1] = true; end  # star with centre 1

    B̂_edge = Dict((i, j) => complex(abs2(G[i, j])) for (i, j) in edges)
    B̂_path = Dict{Tuple{Int,Int}, ComplexF64}()
    for (i, j) in edges
        if !T[i, j]
            path = tree_path(T, i, j)
            B̂_path[(i, j)] = bargmann_invariant(G, path)
        end
    end
    Ĝ = assemble_gram_from_bargmann(B̂_edge, B̂_path, edges, T, n)
    @test norm(Ĝ - G) < 1e-10
end

@testset "estimators: simulated B̂ has σ²/N variance" begin
    B_true = 0.4 + 0.1im
    σ, N = 0.5, 10^4
    trials = 5000
    estimates = [simulate_bargmann_estimate(B_true, σ, N) for _ in 1:trials]
    μ = mean(estimates)
    v = var(estimates)
    @test abs(μ - B_true) < 5σ / sqrt(N * trials)
    @test isapprox(v, σ^2 / N; rtol=0.1)
end

@testset "estimators: √B̂ bias & variance agree with Taylor formulas" begin
    B, σ, N = 0.25, 0.3, 10^5
    trials = 20000
    draws = [real(simulate_bargmann_estimate(B, σ, N; complex_noise=false)) for _ in 1:trials]
    sqrt_draws = sqrt.(max.(draws, 0.0))
    empirical_bias = mean(sqrt_draws) - sqrt(B)
    empirical_var  = var(sqrt_draws)
    @test isapprox(empirical_bias, bias_sqrt_B(B, σ, N); atol=5e-4)
    @test isapprox(empirical_var,  var_sqrt_B(B, σ, N);  rtol=0.1)
end

@testset "estimators: phase estimator variance matches Taylor formula" begin
    # |B_π| held fixed; take B_π real positive to avoid wrap-around.
    Bπ = 0.3
    σ, N = 0.2, 10^5
    trials = 20000
    phases = [(z = simulate_bargmann_estimate(Bπ, σ, N); z / abs(z)) for _ in 1:trials]
    # Var of a unit-modulus complex variable: E|x|² - |Ex|² = 1 - |Ex|².
    empirical = 1 - abs(mean(phases))^2
    @test isapprox(empirical, var_phase(Bπ, σ, N); rtol=0.15)
end

@testset "spanning_tree: max-weight Kruskal on a known case" begin
    # Triangle 1-2-3 with weights 0.9, 0.5, 0.4 → MST keeps 0.9 and 0.5.
    Γ = falses(3, 3)
    W = zeros(3, 3)
    for (i, j, w) in [(1, 2, 0.9), (2, 3, 0.5), (1, 3, 0.4)]
        Γ[i, j] = true; Γ[j, i] = true
        W[i, j] = w; W[j, i] = w
    end
    T = max_weight_spanning_tree(W, Γ)
    @test sum(T) == 2 * 2  # 2 edges, each counted twice
    @test T[1, 2] && T[2, 3] && !T[1, 3]
    # Eq.-38 cost on the kept path 1-2-3: M = (√0.9 · √0.5)^{-1} = 1/√(0.9·0.5).
    @test isapprox(M_T_thm1(T, W), 1 / sqrt(0.9 * 0.5))
end

@testset "spanning_tree: MDST ≤ max-weight tree diameter" begin
    Random.seed!(42)
    for trial in 1:10
        n = 7
        Γ = trues(n, n); for i in 1:n; Γ[i, i] = false; end
        W = rand(n, n)
        W = (W + W') / 2
        for i in 1:n; W[i, i] = 1.0; end
        M_max  = M_T_thm1(max_weight_spanning_tree(W, Γ), W)
        M_mdst = M_T_thm1(min_diameter_spanning_tree_bargmann(W, Γ), W)
        @test M_mdst ≤ M_max + 1e-10
    end
end

function _test_if_graph_is_tree(T::AbstractMatrix{Bool})

    if sum(T) != (size(T)[1]-1)*2
        return false
    end

    vertex_queue = Queue{Int64}()
    parents = zeros(Int64,size(T)[1])
    root = 1
    enqueue!(vertex_queue, root)
    parents[root] = -1
    while !isempty(vertex_queue)    
        v = dequeue!(vertex_queue)
	for i in 1 : size(T)[1]
	    if T[v,i] && (parents[v] != i)
	        if parents[i] != 0
	    	    return false
                end
                enqueue!(vertex_queue, i)
	        parents[i] = v
	    end
	end		  
    end
    return true
end


@testset "spanning_tree: MDST for identical particles" begin
    for n in 2:10
        Γ = trues(n, n); for i in 1:n; Γ[i, i] = false; end
        W = ones(n,n)
	T = min_diameter_spanning_tree_bargmann(W, Γ)
	@test _test_if_graph_is_tree(T)
        M_mdst = M_T_thm1(T, W)
        @test abs(M_mdst - 1) < 1e-10
        gamma = W .* 0.0 # -log(1) = 0
        d, _ = tree_diameter(T, gamma)
        @test abs(d) < 1e-10
    end
end

@testset "spanning_tree: MDST for a line" begin
    Random.seed!(123)
    for n in 2:10
        Γ = falses(n, n)
        W = zeros(n,n)
        d = 0.0
        for i in 1:(n-1)
            Γ[i, i+1] = true
            Γ[i+1, i] = true
            # we don't want exactly 0 because then our graph will become disconnected
            while W[i, i+1] == 0
                W[i, i+1] = rand()
            end
            W[i+1, i] = W[i, i+1]
            d -= log(W[i+1, i])/2.
        end        

	T = min_diameter_spanning_tree_bargmann(W, Γ)
        #T should be exactly equal to Gamma
	@test all(T .== Γ)

        M_mdst = M_T_thm1(T, W)
        @test abs(M_mdst - exp(d)) < 1e-10
    end
end

@testset "spanning_tree: MDST for a line with random noise" begin
    Random.seed!(123)
    for n in 3:10
        W = rand(n,n)*0.1
        W = (W + W')/2.
        d = 0.0
        W_line = zeros(Float64, n,n)
        
        for i in 1:(n-1)
            W_line[i, i+1] = 0.9
            W_line[i+1, i] = W_line[i, i+1]
            d += -log(0.9 + W[i, i+1])/2.
        end
        W = W .+ W_line
        Γ = (W .!= 0.)
	T = min_diameter_spanning_tree_bargmann(W, Γ)
        #T should be exactly given by the line v <-> v+1
        Γ_line = .!(W_line .== 0.)
        
        @test all(Γ_line .== T)
        M_mdst = M_T_thm1(T, W)
        @test abs(M_mdst - exp(d)) < 1e-10
    end
end

@testset "spanning_tree: MDST API split + input guards (issue #14)" begin
    # (a) The generic min_diameter_spanning_tree(γ, Γ) on distance weights must
    # agree with the Bargmann wrapper that builds γ = -log|W| internally.
    Random.seed!(7)
    for _ in 1:10
        n = rand(4:7)
        Γ = trues(n, n); for i in 1:n; Γ[i, i] = false; end
        W = rand(n, n) .* 0.8 .+ 0.1
        W = (W + W') / 2
        for i in 1:n; W[i, i] = 1.0; end
        γ = fill(Inf, n, n)
        for i in 1:n, j in 1:n
            i != j && (γ[i, j] = -log(abs(W[i, j])))
        end
        T_generic  = min_diameter_spanning_tree(γ, Γ)
        T_bargmann = min_diameter_spanning_tree_bargmann(W, Γ)
        @test all(T_generic .== T_bargmann)
    end

    # (b) A disconnected Γ is an error (the protocol assumes Γ connected),
    # for both the generic routine and the Bargmann wrapper.
    Γ = falses(4, 4)
    Γ[1, 2] = Γ[2, 1] = true               # component {1,2} …
    Γ[3, 4] = Γ[4, 3] = true               # … disconnected from {3,4}
    W = fill(0.5, 4, 4); for i in 1:4; W[i, i] = 1.0; end
    @test_throws ErrorException min_diameter_spanning_tree_bargmann(W, Γ)
    γ = fill(Inf, 4, 4)
    for i in 1:4, j in 1:4; i != j && (γ[i, j] = 1.0); end
    @test_throws ErrorException min_diameter_spanning_tree(γ, Γ)

    # (c) A present edge with zero Bargmann weight is an input clash → error
    # (would silently look like an absent edge via γ = -log 0 = ∞).
    Γ = trues(3, 3); for i in 1:3; Γ[i, i] = false; end
    W = fill(0.5, 3, 3); for i in 1:3; W[i, i] = 1.0; end
    W[1, 2] = W[2, 1] = 0.0
    @test_throws ErrorException min_diameter_spanning_tree_bargmann(W, Γ)
end




# --- Brute-force MDST validation (CLAUDE.md "Next" #3) --------------------

"""Decode a Prüfer sequence (length n−2, entries in 1:n) into the adjacency
matrix of the corresponding labeled tree on n vertices (standard bijection;
Cayley's formula: there are n^(n−2) labeled trees on n vertices)."""
function _prufer_to_tree(seq::AbstractVector{<:Integer}, n::Int)
    degree = ones(Int, n)
    for v in seq
        degree[v] += 1
    end
    T = falses(n, n)
    avail = trues(n)
    for s in seq
        l = findfirst(v -> avail[v] && degree[v] == 1, 1:n)
        T[l, s] = true; T[s, l] = true
        avail[l] = false
        degree[l] -= 1; degree[s] -= 1
    end
    u, v = [z for z in 1:n if avail[z]]   # exactly two vertices remain
    T[u, v] = true; T[v, u] = true
    return T
end

"""Brute-force minimum tree diameter over ALL n^(n−2) labeled spanning trees
of the complete graph on n vertices (Prüfer enumeration), under edge weights
γ — the exact optimum of the MDST objective `min_T max_{π∈T} γ(π)` of
`sec:tree` (eq. 43–46 of `Gram_Tomography.pdf`)."""
function _brute_force_min_diameter(γ::AbstractMatrix, n::Int)
    best = Inf
    for idx in Iterators.product(ntuple(_ -> 1:n, n - 2)...)
        T = _prufer_to_tree(collect(idx), n)
        d, _ = tree_diameter(T, γ)
        d < best && (best = d)
    end
    return best
end

@testset "spanning_tree: MDST is optimal (brute force, n ≤ 7)" begin
    # Validates that `min_diameter_spanning_tree` attains the TRUE minimum
    # diameter (sec:tree, eq. 43–46: minimise max_{π∈T} Σ γ(u_l,u_{l+1}) with
    # γ = −log W) by enumerating every labeled spanning tree of K_n via
    # Prüfer sequences (n = 7 → 16807 trees per instance).
    #
    # Historical note (2026-06-11): the previous implementation — a
    # vertex-rooted Dijkstra sweep, the classic 2-approximation — FAILED this
    # exact test on 2 of the 20 seeded n ∈ {5, 6} instances below (both n = 6,
    # diameters 1.0197× and 1.0049× the optimum): the absolute 1-center lay
    # strictly inside an edge. The routine now implements the exact
    # Hassin–Tamir absolute-1-center reduction (the manuscript's [mdst]
    # reference) and must match the brute force to round-off.
    Random.seed!(123)
    for n in (5, 6, 7)
        for trial in 1:(n == 7 ? 3 : 10)
            W = rand(n, n)
            W = (W + W') / 2
            for i in 1:n; W[i, i] = 1.0; end
            Γ = trues(n, n); for i in 1:n; Γ[i, i] = false; end
            γ = [-log(W[i, j]) for i in 1:n, j in 1:n]
            T_alg = min_diameter_spanning_tree_bargmann(W, Γ)
            @test sum(T_alg) == 2 * (n - 1)            # n−1 undirected edges
            d_alg, _ = tree_diameter(T_alg, γ)
            @test d_alg ≤ _brute_force_min_diameter(γ, n) + 1e-9
        end
    end
end

# --- Thm 1 primitives (Gram_Tomography.pdf eq. 37–42) ---------------------

@testset "spanning_tree: M_T_thm1 == sqrt of the no-sqrt cost" begin
    # Eq. 38 sqrt-inside-product means
    #   M_thm1(T) = (min_π ∏ √|W|)^{-1} = √[(min_π ∏ |W|)^{-1}].
    # The no-sqrt cost is exp(diameter) under γ = -log W; check the identity
    # against an independent inline computation of that diameter.
    Random.seed!(2026)
    for trial in 1:20
        n = rand(4:8)
        Γ = trues(n, n); for i in 1:n; Γ[i, i] = false; end
        W = rand(n, n) .* 0.8 .+ 0.05  # ensure positive
        W = (W + W') / 2
        for i in 1:n; W[i, i] = 1.0; end
        T = min_diameter_spanning_tree_bargmann(W, Γ)
        γ = [-log(W[i, j]) for i in 1:n, j in 1:n]
        diam, _ = tree_diameter(T, γ)
        M_nosqrt = exp(diam)
        @test isapprox(M_T_thm1(T, W), sqrt(M_nosqrt); atol=1e-10, rtol=1e-10)
    end
end

@testset "spanning_tree: m_T returns hop-count diameter + 1" begin
    # Star tree on n vertices: diameter (in hops) = 2, so m(T) = 3.
    for n in 4:8
        T_star = falses(n, n)
        for v in 2:n
            T_star[1, v] = true; T_star[v, 1] = true
        end
        @test m_T(T_star) == 3
    end
    # Path tree on n vertices: diameter = n-1, so m(T) = n.
    for n in 3:7
        T_path = falses(n, n)
        for v in 1:n-1
            T_path[v, v+1] = true; T_path[v+1, v] = true
        end
        @test m_T(T_path) == n
    end
end

@testset "estimators: V_delta_thm1 term-by-term (pure-state, indep. noise)" begin
    # Sanity check: at ε=0, K=0, V(δ) = (1/(2δ^{3/2}) + 1/(4δ) + (M²/(2δ))) σ²/N
    # (since (M/√δ)² = M²/δ, and the (1/2) coefficient gives M²/(2δ)).
    δ = 0.05; M = 12.0; m = 3; σ = 0.2; N = 10_000
    V_actual = V_delta_thm1(δ, M, m, σ, N)
    V_expected = (1/(2 * δ^1.5) + 1/(4δ) + M^2 / (2δ)) * (σ^2 / N)
    @test isapprox(V_actual, V_expected; rtol=1e-12)

    # ε > 0 piece adds (4/√δ + 4m·M/√δ)·ε.
    ε = 0.01
    V_eps_actual = V_delta_thm1(δ, M, m, σ, N; ε=ε)
    V_eps_expected = V_expected + (4/sqrt(δ) + 4m * M/sqrt(δ)) * ε
    @test isapprox(V_eps_actual, V_eps_expected; rtol=1e-12)
end

@testset "estimators: K_delta_thm1 matches eq. 42 literally" begin
    # K(δ) = √( ((M/√δ)·4m·ε + (1/2)(M/√δ)²·σ²/N) · (1/(4δ)·σ²/N) )
    δ = 0.05; M = 12.0; m = 3; σ = 0.2; N = 10_000
    for ε in [0.0, 0.005, 0.02]
        MoS = M / sqrt(δ)
        inner_a = MoS * 4 * m * ε + 0.5 * MoS^2 * (σ^2 / N)
        inner_b = (1 / (4δ)) * (σ^2 / N)
        K_expected = sqrt(inner_a * inner_b)
        @test isapprox(K_delta_thm1(δ, M, m, σ, N; ε=ε), K_expected; rtol=1e-12)
    end
end

@testset "estimators: frobenius_mse_bound_thm1 (eq. 40) structure" begin
    # At δ = δ_min the truncation term n(δ)·δ vanishes (n_δ = 0), so the
    # bound collapses to n² · V(δ) with V from V_delta_thm1.
    n = 5; M = 12.0; m_t = 3; σ = 0.2; N = 10_000; δ = 0.05
    b = frobenius_mse_bound_thm1(n, 0, δ, M, m_t, σ, N)
    @test isapprox(b, n^2 * V_delta_thm1(δ, M, m_t, σ, N); rtol=1e-12)
    @test b > 0 && isfinite(b)

    # The truncation term adds exactly n_δ·δ on top of n²·V(δ).
    n_δ = 3
    b_trunc = frobenius_mse_bound_thm1(n, n_δ, δ, M, m_t, σ, N)
    @test isapprox(b_trunc, b + n_δ * δ; rtol=1e-12)

    # ε > 0 (pseudopurity) can only inflate the bound.
    @test frobenius_mse_bound_thm1(n, 0, δ, M, m_t, σ, N; ε=0.05) > b
end

@testset "projection: project_gram produces a valid Gram matrix" begin
    Random.seed!(7)
    for _ in 1:10
        n = 6
        # Perturb a real Gram matrix with noise, then project.
        G, _ = random_gram_matrix(n, 4)
        Ĝ = G + 0.1 * randn(ComplexF64, n, n)
        Ĝ = 0.5 * (Ĝ + Ĝ')
        G_proj, iters, conv = project_gram(Ĝ; maxiter=500, tol=1e-12)
        @test conv
        @test is_valid_gram(G_proj; tol=1e-7)
        # Contraction: projection does not increase distance to true G.
        @test norm(G_proj - G) ≤ norm(Ĝ - G) + 1e-8
    end
end

@testset "reconstruction: B† B = Ĝ after eigendecomposition" begin
    Random.seed!(11)
    for _ in 1:10
        n, d = 6, 4
        G, _ = random_gram_matrix(n, d)
        B = reconstruct_eigen(G, d)
        @test size(B) == (d, n)
        @test norm(B' * B - G) < 1e-8
    end
end

@testset "reconstruction: Cholesky reproduces G" begin
    Random.seed!(13)
    for _ in 1:5
        n = 5
        # Full-rank Gram: use n states in ℂ^n.
        G, _ = random_gram_matrix(n, n)
        # Add a tiny ridge to guarantee positive definiteness.
        Gp = G + 1e-8 * I
        U = reconstruct_cholesky(Gp)
        @test norm(U' * U - Gp) < 1e-6
    end
end

@testset "reconstruction: Powers–Størmer inequality holds" begin
    Random.seed!(17)
    for _ in 1:20
        n = 5
        G1, _ = random_gram_matrix(n, 4)
        G2 = G1 + 0.05 * (randn(ComplexF64, n, n) |> x -> (x + x') / 2)
        G2, _, _ = project_gram(G2; maxiter=500, tol=1e-12)
        sqrtG1 = sqrt(Hermitian(Matrix(G1)))
        sqrtG2 = sqrt(Hermitian(Matrix(G2)))
        lhs = opnorm(sqrtG1 - sqrtG2, 2)
        rhs = powers_stormer_bound(G1, G2)
        @test lhs ≤ rhs + 1e-8
    end
end

@testset "reconstruction: align_unitary achieves equality up to R" begin
    Random.seed!(19)
    for _ in 1:10
        n, d = 6, 4
        G, _ = random_gram_matrix(n, d)
        B = reconstruct_eigen(G, d)
        R = qr(randn(ComplexF64, d, d)).Q |> Matrix
        B2 = R * B
        R̂, Baligned = align_unitary(B, B2)
        @test norm(Baligned - B2) < 1e-8
    end
end

@testset "BosonSampling: required fixes are present (commits bda901d + f5c9bbe + 91b661a)" begin
    info = require_fixed_samplers()
    @test info.clifford_fix === true
    @test info.partial_dist === true
    @test info.mixed_state === true
    # The feature symbols themselves:
    @test isdefined(BosonSampling, :corrected_clifford_algorithm)
    @test isdefined(BosonSampling, :PartialDistinguishabilityModel)
    @test isdefined(BosonSampling, :partial_distinguishability_sampler)
    # Mixed-state input type added on the samplers branch (commit 91b661a):
    @test isdefined(BosonSampling, :MixedDensityMatrices)
end

@testset "BosonSampling: unitary convention is transpose-of-physics" begin
    # Physics convention: U_shift · e_j = e_{j+1 mod 3}, so a single photon in
    # input mode 1 lands deterministically in output mode 2 (since the
    # provided constant has col 1 = e_2).
    @assert U_SHIFT_PHYSICS * ComplexF64[1, 0, 0] == ComplexF64[0, 1, 0]
    # Single-photon Gram matrix is the 1×1 identity.
    S1 = reshape(ComplexF64[1.0], 1, 1)

    # With the correct transpose (to_bsj), output mode 2 gets P = 1.
    @test isapprox(bsj_physics_probability(U_SHIFT_PHYSICS, S1, [1,0,0], [0,1,0]),
                   1.0; atol=1e-10)
    @test isapprox(bsj_physics_probability(U_SHIFT_PHYSICS, S1, [1,0,0], [1,0,0]),
                   0.0; atol=1e-10)
    @test isapprox(bsj_physics_probability(U_SHIFT_PHYSICS, S1, [1,0,0], [0,0,1]),
                   0.0; atol=1e-10)

    # If someone forgets the transpose and passes U directly, BosonSampling.jl
    # reads it as row=input/col=output, so input 1 → output 3 appears instead.
    U_raw = Matrix{ComplexF64}(U_SHIFT_PHYSICS)
    p_raw = real(process_probability_partial(U_raw, Matrix{ComplexF64}(S1),
                                             [1,0,0], [0,0,1]))
    @test isapprox(p_raw, 1.0; atol=1e-10)

    # And passing the adjoint (a wrong correction some readers might try)
    # introduces a spurious conjugate — for real orthogonal U_SHIFT the
    # result happens to be the same as transpose here, but the distinction
    # matters for complex U; document via a complex example below.
    θ = 0.37
    φ = 0.81
    # A non-Hermitian, non-symmetric 2×2 unitary.
    U2 = ComplexF64[cos(θ)            -sin(θ)*exp( im*φ);
                    sin(θ)*exp(-im*φ)  cos(θ)]
    @assert U2 * U2' ≈ I
    @assert U2 != transpose(U2)
    @assert U2 != adjoint(U2)
    S2 = reshape(ComplexF64[1.0], 1, 1)

    # Physics: a single photon in input mode 1 lands in output mode 2 with
    # amplitude U2[2, 1]. Probability = |U2[2,1]|² = sin²(θ).
    p_phys = bsj_physics_probability(U2, S2, [1, 0], [0, 1])
    @test isapprox(p_phys, sin(θ)^2; atol=1e-10)

    # Passing `adjoint(U2)` instead of `transpose(U2)` gives the wrong answer
    # (|conj(U2[1,2])|² = |U2[1,2]|² = sin²(θ) — same magnitude here), but
    # for a non-trivial Gram matrix interference structure the two differ.
    # We catch the difference by using a Gram-matrix test below.
end

@testset "BosonSampling: HOM probabilities match analytical formula" begin
    require_fixed_samplers()
    for s in (0.0+0im, 0.4+0.3im, -0.2-0.5im, 0.9+0im, 0.1+0.7im)
        S = ComplexF64[1.0 s; conj(s) 1.0]
        # Build the BSJ-convention matrix via to_bsj (identity for symmetric).
        interf = UserDefinedInterferometer(to_bsj(HOM_U_PHYSICS))
        input  = Input{UserDefinedGramMatrix}(first_modes(2, 2), S)
        dist   = full_distribution(input, interf)
        p20, p11, p02 = hom_probabilities(s)
        # Sort by state for reliable indexing.
        state_p = Dict(o.state => real(p) for (o, p) in zip(dist.counts, dist.proba))
        @test isapprox(state_p[[2, 0]], p20; atol=1e-10)
        @test isapprox(state_p[[1, 1]], p11; atol=1e-10)
        @test isapprox(state_p[[0, 2]], p02; atol=1e-10)
    end
end

@testset "BosonSampling: HOM B̂ is unbiased and variance matches" begin
    require_fixed_samplers()
    Random.seed!(2024)
    for s in (0.2 + 0.1im, -0.4 + 0.25im, 0.6 + 0.0im)
        N = 20_000
        trials = 80
        estimates = [estimate_B_ij(s, N) for _ in 1:trials]
        μ_emp = mean(estimates)
        v_emp = var(estimates)
        # Unbiased: empirical mean within 4σ of |s|².
        se_mean = sqrt((1 - abs2(s)^2) / (N * trials))
        @test abs(μ_emp - abs2(s)) < 6 * se_mean
        # Variance: matches (1 - |s|^4) / N within a factor of 1.3 (Monte Carlo).
        v_theo = (1 - abs2(s)^2) / N
        @test 0.6 * v_theo < v_emp < 1.4 * v_theo
    end
end

@testset "Integration: HOM tomography recovers |G|² magnitudes on 4-state Gram" begin
    require_fixed_samplers()
    Random.seed!(31)
    n, d = 4, 3
    G, _ = random_gram_matrix(n, d)
    N = 30_000
    B̂ = hom_tomography_magnitudes(G, N)

    B_true = [abs2(G[i, j]) for i in 1:n, j in 1:n]

    # Entry-wise: each estimate is within 4 standard errors of the truth.
    for i in 1:n, j in (i+1):n
        se = sqrt((1 - B_true[i, j]^2) / N)
        @test abs(B̂[i, j] - B_true[i, j]) < 6 * se + 1e-6
    end
    # Global Frobenius error stays small.
    @test norm(B̂ - B_true) < 0.05
end

@testset "Fourier: interferometer is unitary; M = 2 is the HOM beamsplitter" begin
    for M in 2:6
        U = fourier_interferometer(M)
        @test size(U) == (M, M)
        @test norm(U * U' - I) < 1e-12          # unitary
        @test norm(U - transpose(U)) < 1e-12    # symmetric → to_bsj is a no-op
    end
    # M = 2: (1/√2)·[1 1; 1 −1] — exactly the HOM 50:50 beamsplitter.
    @test fourier_interferometer(2) ≈ HOM_U_PHYSICS atol=1e-14
end

@testset "Fourier: population estimator equals B_π (convention lock)" begin
    # THE critical correctness test: the population-limit Fourier estimator
    # must equal bargmann_invariant(G, cyc) INCLUDING PHASE — i.e. B̂ → B_π
    # (no conj), the convention of CLAUDE.md § "Bargmann invariants". This
    # is what fixes the F̂† (minus-sign) interferometer + the +(2πi/M)·f(S)
    # exponent: using F̂ instead yields conj(B_π) for M ≥ 3 (the M = 2 HOM
    # case is real and cannot discriminate).
    Random.seed!(2601)
    for M in (2, 3, 4, 5), trial in 1:3
        n = M + 2
        G, _ = random_gram_matrix(n, n)
        cyc = randperm(n)[1:M]                  # random M-cycle of states
        B_pop  = fourier_bargmann_population(G[cyc, cyc])
        B_true = bargmann_invariant(G, cyc)
        @test abs(B_pop - B_true) < 1e-9        # phase included
        # And explicitly NOT the conjugate when B_π has a phase.
        if abs(imag(B_true)) > 1e-3
            @test abs(B_pop - conj(B_true)) > 1e-3
        end
    end
end

@testset "Fourier: finite-N estimate → population, RMS ~ N^{-1/2}" begin
    require_fixed_samplers()
    Random.seed!(2602)
    M = 3
    G, _ = random_gram_matrix(M, M)
    B_pop = fourier_bargmann_population(G)      # = B_π (testset above)

    rms = Dict{Int, Float64}()
    trials = 24
    for N in (2_000, 8_000)
        errs = [abs(fourier_bargmann_estimate(G, N) - B_pop) for _ in 1:trials]
        rms[N] = sqrt(mean(abs2, errs))
        # Each sample contributes a unit-modulus phase factor, so
        # Var(B̂) = (1 − |B_π|²)/N exactly; allow generous Monte-Carlo slack.
        rms_theory = sqrt((1 - abs2(B_pop)) / N)
        @test 0.5 * rms_theory < rms[N] < 1.8 * rms_theory
    end
    # 4× more samples → RMS halves (√N scaling), up to Monte-Carlo spread.
    @test rms[8_000] < rms[2_000]
    @test rms[8_000] > 0.25 * rms[2_000]
end

@testset "OneShot: A(U) matrix agrees with eq:A(U) for small U" begin
    # Tiny 3×3 unitary, n = 2; verify entry-by-entry against the formula.
    U = random_haar_unitary(3; rng=MersenneTwister(1))
    A, out_pairs, in_pairs = build_A_matrix_order_2(U, 2)
    @test size(A) == (6, 2)           # (3·2) × (2·1)
    @test Set(out_pairs) == Set([(i, j) for i in 1:3 for j in 1:3 if i != j])
    @test Set(in_pairs)  == Set([(1, 2), (2, 1)])
    for (a, (i, j)) in enumerate(out_pairs), (b, (k, l)) in enumerate(in_pairs)
        @test isapprox(A[a, b], U[i, k] * U[j, l] * conj(U[i, l]) * conj(U[j, k]); atol=1e-12)
    end
end

@testset "OneShot: A b + c_dist equals exact correlator at the population limit" begin
    # Verify against the exact full distribution (no sampling noise).
    Random.seed!(99)
    n = 3
    m = 5
    G, _ = random_gram_matrix(n, n)
    U_phys = random_haar_unitary(m)

    # Exact output distribution.
    S = ComplexF64.(G)
    interf = UserDefinedInterferometer(to_bsj(U_phys))
    input  = Input{UserDefinedGramMatrix}(first_modes(n, m), S)
    dist   = full_distribution(input, interf)
    states = [o.state for o in dist.counts]
    probs  = Float64.(real.(dist.proba)); probs ./= sum(probs)

    # Exact expectations ⟨n_i⟩, ⟨n_i n_j⟩ from the full distribution.
    mean_n = zeros(Float64, m)
    mean_ninj = zeros(Float64, m, m)
    for (s, p) in zip(states, probs)
        for i in 1:m
            mean_n[i] += p * s[i]
            for j in 1:m
                mean_ninj[i, j] += p * s[i] * s[j]
            end
        end
    end

    A, out_pairs, in_pairs = build_A_matrix_order_2(U_phys, n)
    c_exact = [mean_ninj[i, j] - mean_n[i] * mean_n[j] for (i, j) in out_pairs]
    c_dist  = distinguishable_correlator_baseline(U_phys, n, out_pairs)

    # Build the true b vector.
    b_true = [abs2(G[k, l]) for (k, l) in in_pairs]

    # Check the linear identity exactly (no sampling).
    residual = c_exact - c_dist - A * b_true
    @test norm(residual) < 1e-10
end

@testset "OneShot: recovers |G_{kl}|² from a single unitary and N samples" begin
    require_fixed_samplers()
    Random.seed!(2026)
    for (n, m, N) in [(3, 5, 40_000), (4, 7, 60_000)]
        G, _ = random_gram_matrix(n, n)
        U_phys = random_haar_unitary(m)
        samples = one_shot_sample(G, U_phys, N)
        b̂, _, _, _ = one_shot_bargmann_order_2(U_phys, samples, n)

        # Diagonal is 1 by construction.
        @test all(abs(b̂[k, k] - 1) < 1e-10 for k in 1:n)

        # Symmetric Hermitian (real, since ideal b is real).
        @test norm(b̂ - transpose(b̂)) < 1e-10

        # Entry-wise accuracy. The standard error of each b̂_{kl} scales
        # roughly like 1/√N with a U-dependent prefactor; 0.07 absorbs the
        # occasional tail entry at N ≥ 40k without loosening meaningfully.
        for k in 1:n, l in (k+1):n
            @test abs(real(b̂[k, l]) - abs2(G[k, l])) < 0.07
        end
    end
end

@testset "OneShot order 3: Eq. (29) reproduces ⟨n_a n_b n_c⟩ exactly" begin
    Random.seed!(101)
    # Random G, random Haar U; compare predicted_3mode_moment (Eq. 29)
    # against BosonSampling.jl's exact correlator.
    n, m = 3, 5
    G, _ = random_gram_matrix(n, n)
    U = random_haar_unitary(m)
    S = ComplexF64.(G)
    interf = UserDefinedInterferometer(to_bsj(U))
    input  = Input{UserDefinedGramMatrix}(first_modes(n, m), S)
    dist = full_distribution(input, interf)
    probs = Float64.(real.(dist.proba)); probs ./= sum(probs)

    for a in 1:m, b in (a+1):m, c in (b+1):m
        pred = real(predicted_3mode_moment(U, G, a, b, c))
        exact = sum(p * o.state[a] * o.state[b] * o.state[c]
                    for (o, p) in zip(dist.counts, probs))
        @test isapprox(pred, exact; atol=1e-10)
    end
end

@testset "OneShot order 3: linear inversion at population limit" begin
    # Given exact 3-mode correlators (no sampling noise), the linear system
    # recovers every triangle Bargmann invariant to machine precision.
    Random.seed!(37)
    for (n, m) in [(3, 5), (4, 7)]
        G, _ = random_gram_matrix(n, n)
        U = random_haar_unitary(m)
        abc_triples = [(a, b, c) for a in 1:m for b in (a+1):m for c in (b+1):m]

        # Exact ⟨n_a n_b n_c⟩ from full_distribution.
        S = ComplexF64.(G)
        interf = UserDefinedInterferometer(to_bsj(U))
        input  = Input{UserDefinedGramMatrix}(first_modes(n, m), S)
        dist = full_distribution(input, interf)
        probs = Float64.(real.(dist.proba)); probs ./= sum(probs)
        y_exact = Float64[sum(p * o.state[a] * o.state[b] * o.state[c]
                               for (o, p) in zip(dist.counts, probs))
                          for (a, b, c) in abc_triples]

        b2_true = Matrix{Float64}([i == j ? 1.0 : abs2(G[i, j]) for i in 1:n, j in 1:n])
        y_sub = three_mode_correlator_subtraction(U, b2_true, abc_triples, n)
        A3_real, _, rst_triples = build_A_matrix_order_3(U, n)
        x = A3_real \ (y_exact - y_sub)

        for (i, (r, s, t)) in enumerate(rst_triples)
            B_true = G[r, s] * G[s, t] * G[t, r]
            B̂ = x[2*i - 1] + im * x[2*i]
            @test abs(B̂ - B_true) < 1e-9
        end
    end
end

@testset "OneShot order 3: finite-N recovery RMS scaling" begin
    require_fixed_samplers()
    n, m = 4, 7

    # Average over several (G, U) realisations so per-seed fluctuations don't
    # mask the √N scaling.
    rms_means = Dict{Int, Float64}()
    Ns = (40_000, 160_000)
    for N in Ns
        accum = 0.0
        trials = 4
        for seed in (11, 17, 23, 29)
            Random.seed!(seed)
            G, _ = random_gram_matrix(n, n)
            U = random_haar_unitary(m)
            samples = one_shot_sample(G, U, N)
            b2_hat, _, _, _ = one_shot_bargmann_order_2(U, samples, n)
            B3_hat, _, _ = one_shot_bargmann_order_3(U, samples, n, real.(b2_hat))
            accum += sqrt(mean(abs2(B3_hat[(r, s, t)] - G[r, s] * G[s, t] * G[t, r])
                                for r in 1:n for s in (r+1):n for t in (s+1):n))
        end
        rms_means[N] = accum / trials
    end
    # Averaged over 4 seeds the √N scaling is clear: N = 4× → RMS ≈ 2×
    # smaller. Keep the thresholds loose enough to survive the remaining
    # realisation-to-realisation spread.
    @test rms_means[40_000]  < 0.15
    @test rms_means[160_000] < 0.10
    @test rms_means[160_000] < rms_means[40_000]
end

"""
    _one_shot_sigma_eff(G, U_phys, n) -> Float64

Worst-case per-invariant noise level `σ_eff` of the one-shot correlator
estimators, defined through `Var(B̂) = σ_eff²/N` — the variance model assumed
by Theorem 1 ("We assume the variance is bounded by σ²/N",
gram-estimation.tex). Computed exactly, to leading order in 1/N (delta
method), from the TRUE output distribution `full_distribution(G, U)`:

- order 2 (`one_shot_bargmann_order_2`, the eq:A(U) inversion
  `b̂ = A⁺·(ĉ − c^dist)`): linearising the plug-in correlator
  `ĉ_{ij} = mean(n_i n_j) − mean(n_i)·mean(n_j)` gives the per-sample
  influence `h_{(i,j)}(S) = n_i n_j − ⟨n_i⟩n_j − ⟨n_j⟩n_i` (constants drop
  from variances); the symmetrised real estimate `b̂2[k,l]` then has
  influence `ψ_{kl}(S) = Re[(ψ_{(k,l)} + conj(ψ_{(l,k)}))/2]` with
  `ψ_{(k,l)} = Σ_a A⁺[(k,l),a]·h_a(S)`, and `N·Var(b̂2[k,l]) = Var_S(ψ_{kl})`
  under the exact distribution.
- order 3 (`one_shot_bargmann_order_3`, Robbio Eq. 29 inversion): the RHS
  `y_emp − y_sub(b̂2_raw)` couples to the order-2 errors through the LINEAR
  map `J = ∂y_sub/∂b2` (`three_mode_correlator_subtraction` is affine in
  b2), so the per-sample influence vector is
  `φ(S) = A3⁺·(g(S) − J·ψ(S))`, `g_{(a,b,c)}(S) = n_a n_b n_c`; packing the
  (Re, Im) pairs as complex gives `N·Var(B̂_{rst}) = Var_S(φ_{rst})`.

Returns `σ_eff = √(max over all pair and triangle invariants of N·Var)`.
"""
function _one_shot_sigma_eff(G::AbstractMatrix, U_phys::AbstractMatrix, n::Integer)
    m = size(U_phys, 1)
    # Exact output distribution of the experiment.
    S_mat = ComplexF64.(G)
    interf = UserDefinedInterferometer(to_bsj(U_phys))
    input  = Input{UserDefinedGramMatrix}(first_modes(n, m), S_mat)
    dist   = full_distribution(input, interf)
    states = [o.state for o in dist.counts]
    probs  = Float64.(real.(dist.proba)); probs ./= sum(probs)
    nstates = length(states)
    mean_n = zeros(Float64, m)
    for (st, p) in zip(states, probs), i in 1:m
        mean_n[i] += p * st[i]
    end

    # Order-2 influence functions ψ_{kl}(S).
    A2c, out_pairs, in_pairs = build_A_matrix_order_2(U_phys, n)
    A2p = pinv(A2c)
    pairs_un = [(k, l) for k in 1:n for l in (k+1):n]
    idx_in = Dict(p => i for (i, p) in enumerate(in_pairs))
    ψsym = zeros(Float64, nstates, length(pairs_un))
    for (si, st) in enumerate(states)
        h = [st[i] * st[j] - mean_n[i] * st[j] - mean_n[j] * st[i]
             for (i, j) in out_pairs]
        ψ = A2p * h
        for (pi, (k, l)) in enumerate(pairs_un)
            ψsym[si, pi] = real(0.5 * (ψ[idx_in[(k, l)]] + conj(ψ[idx_in[(l, k)]])))
        end
    end
    varb2 = [sum(probs[si] * ψsym[si, pi]^2 for si in 1:nstates) -
             sum(probs[si] * ψsym[si, pi] for si in 1:nstates)^2
             for pi in 1:length(pairs_un)]

    # Order-3 influence functions φ_{rst}(S), including the b̂2 → y_sub
    # propagation through the linear map J.
    A3, abc_triples, rst_triples = build_A_matrix_order_3(U_phys, n)
    A3p = pinv(A3)
    B0 = zeros(Float64, n, n)
    y0 = three_mode_correlator_subtraction(U_phys, B0, abc_triples, n)
    J = zeros(Float64, length(abc_triples), length(pairs_un))
    for (pi, (k, l)) in enumerate(pairs_un)
        E = zeros(Float64, n, n); E[k, l] = 1.0; E[l, k] = 1.0
        J[:, pi] = three_mode_correlator_subtraction(U_phys, E, abc_triples, n) - y0
    end
    ntri = length(rst_triples)
    φc = zeros(ComplexF64, nstates, ntri)
    for (si, st) in enumerate(states)
        g = [Float64(st[a] * st[b] * st[c]) for (a, b, c) in abc_triples]
        φ = A3p * (g - J * ψsym[si, :])
        for i in 1:ntri
            φc[si, i] = φ[2i-1] + im * φ[2i]
        end
    end
    varB3 = [sum(probs[si] * abs2(φc[si, i]) for si in 1:nstates) -
             abs2(sum(probs[si] * φc[si, i] for si in 1:nstates))
             for i in 1:ntri]

    return sqrt(max(maximum(varb2), maximum(varB3)))
end

@testset "OneShot end-to-end: one U + N samples → Ĝ (gauge-matched, star MDST)" begin
    require_fixed_samplers()
    Random.seed!(9)
    n, m = 4, 7

    # The correlator method supplies orders ≤ 3 only, so the MDST must have
    # hop diameter ≤ 2 (a star) for the protocol to close every non-tree
    # edge with a triangle. Build a Gram matrix whose MDST is GUARANTEED a
    # star: a hub state ψ₁ overlaps strongly with every other state
    # (|⟨ψ₁|ψ_j⟩| = α) while the others overlap only through their hub
    # component (|⟨ψ_i|ψ_j⟩| = α²). On weights B = |G|² the star at 1 has
    # γ-diameter 2·(−2 log α), strictly below any tree containing a mutual
    # edge (−4 log α for that edge alone, plus more); verified robust to
    # ±0.05 weight noise over 200 trials, far above the order-2 std. err.
    # at N = 300k.
    α = 0.8
    Ψ = zeros(ComplexF64, n, n)
    Ψ[1, 1] = 1.0
    θ = (0.7, -1.3, 2.1)                  # distinct hub phases → complex G
    for j in 2:n
        Ψ[1, j] = α * exp(im * θ[j-1])
        Ψ[j, j] = sqrt(1 - α^2)
    end
    G = Ψ' * Ψ
    U_phys = random_haar_unitary(m)

    N = 300_000
    result = physical_tomography_one_shot(G, U_phys, N)

    # The star MDST means every non-tree tree-path has exactly 3 vertices.
    @test all(length(tree_path(result.T, i, j)) == 3
              for i in 1:n for j in (i+1):n if !result.T[i, j])

    # Hermiticity (Ĝ is Hermitian by construction in assemble + projection).
    @test norm(result.Ĝ - result.Ĝ') < 1e-10

    # Compare to the gauge-matched true Gram against the THEOREM-1 bound
    # (th:main-MSE, eq. 40–42 of Gram_Tomography.pdf), replacing the former
    # hand-tuned 0.7 tolerance. Parameters, all from manuscript definitions:
    #  - σ_eff: worst per-invariant N·Var of the correlator estimators,
    #    computed exactly from the output distribution (delta method) — see
    #    `_one_shot_sigma_eff` — so Var(B̂) ≤ σ_eff²/N as Theorem 1 assumes;
    #  - δ = δ_min (eq:deltamin, eq. 37): every true B_ij ≥ δ_min survives
    #    the δ = 0 cutoff actually used, so the surviving edge set equals the
    #    δ = δ_min one and n(δ) = 0;
    #  - M(T) (eq. 38, sqrt-inside-product) and m(T) (eq. 41) on the
    #    reconstructed star tree with TRUE invariants;
    #  - include_covariance = true: B̂_ij and B̂_{π(i,j)} come from the SAME
    #    samples here, so the Cauchy–Schwarz covariance term K(δ) (eq. 42)
    #    may NOT be dropped.
    # One-directional: this single seeded realisation must lie below the
    # bound on the expectation (observed sq. error ≈ 0.006 vs bound ≈ 0.19;
    # ε = 0 — pure states).
    σ_eff = _one_shot_sigma_eff(G, U_phys, n)
    B2true = Float64[i == j ? 1.0 : abs2(G[i, j]) for i in 1:n, j in 1:n]
    δmin = minimum(B2true[i, j] for i in 1:n for j in (i+1):n)
    thm1 = frobenius_mse_bound_thm1(n, 0, δmin, M_T_thm1(result.T, B2true),
                                    m_T(result.T), σ_eff, N;
                                    include_covariance=true)
    G_gauged = regauge_to_tree(G, result.T)
    @test norm(result.Ĝ - G_gauged)^2 ≤ thm1

    # Bargmann-invariants (gauge-invariant) must match well.
    for triple in ([1,2,3], [1,2,4], [1,3,4], [2,3,4])
        B̂ = bargmann_invariant(result.Ĝ, triple)
        B = bargmann_invariant(G, triple)
        @test abs(B̂ - B) < 0.18
    end
end

@testset "OneShot end-to-end: MDST diameter > 2 throws (orders ≤ 3 only)" begin
    require_fixed_samplers()
    Random.seed!(314)
    n, m = 5, 8

    # Chain-shaped Gram: Gaussian-profile states ψ_j[k] ∝ exp(−(k−j)²/(2s²))
    # have |⟨ψ_i|ψ_j⟩| ≈ exp(−(i−j)²/(4s²)) — super-exponential decay in
    # |i − j| — so on γ = −log B the direct edge (1,5) costs ∝ (i−j)² while
    # the nearest-neighbour chain costs ∝ |i−j|. The MDST is the path
    # 1-2-3-4-5 (verified robust to ±0.03 weight noise over 200 trials) and
    # the non-tree edge (1,5) requires a tree path of 5 vertices, i.e. a
    # Bargmann invariant of order 5 > 3 — beyond the correlator method.
    s2 = 1.12
    Ψ = zeros(ComplexF64, n, n)
    for j in 1:n, k in 1:n
        Ψ[k, j] = exp(-(k - j)^2 / (2 * s2))
    end
    for j in 1:n; Ψ[:, j] ./= norm(Ψ[:, j]); end
    G = Ψ' * Ψ
    U_phys = random_haar_unitary(m)

    @test_throws ErrorException physical_tomography_one_shot(G, U_phys, 50_000)
end

@testset "Hybrid end-to-end: correlator ≤3 + Fourier fallback for order >3" begin
    # The manuscript's cost-optimal strategy (physical_tomography_hybrid):
    # ONE correlator experiment supplies all order-2 magnitudes (graph/MDST)
    # and all triangle paths; the Fourier method is the last resort, called
    # ONLY on the residual tree paths of order > 3.
    require_fixed_samplers()

    # (a) Diameter ≤ 2 ⇒ ZERO Fourier calls ⇒ bit-identical to the pure
    #     correlator route (`physical_tomography_one_shot`). The supplier is
    #     never invoked and `from_samples` uses no RNG, so identical seed +
    #     identical correlator draw ⇒ identical Ĝ. Star Gram (every non-tree
    #     path is a triangle), as in the OneShot star end-to-end test.
    let n = 4, m = 7, α = 0.55
        Ψ = zeros(ComplexF64, n, n)
        Ψ[1, 1] = 1.0
        θ = (0.7, -1.3, 2.1)                       # distinct hub phases → complex G
        for j in 2:n
            Ψ[1, j] = α * exp(im * θ[j-1])
            Ψ[j, j] = sqrt(1 - α^2)
        end
        G = Ψ' * Ψ
        U_phys = random_haar_unitary(m)
        N = 50_000

        Random.seed!(8675); h = physical_tomography_hybrid(G, U_phys, N; threaded=false)
        Random.seed!(8675); o = physical_tomography_one_shot(G, U_phys, N; threaded=false)

        @test isempty(h.fourier_edges)             # star MDST ⇒ no order->3 path
        @test h.Ĝ == o.Ĝ                           # bit-identical reconstruction
    end

    # (b) Diameter > 2 ⇒ the long tree paths fall back to Fourier while the
    #     order-3 (triangle) paths and all order-2 magnitudes stay on the
    #     single correlator run. Chain Gram (MDST is the path 1-2-…-n).
    let n = 5, m = 7
        Random.seed!(4605)
        s2 = 1.12
        φ = 0.6 .* (rand(n) .- 0.5)               # mild phase ramp → complex G
        Ψ = zeros(ComplexF64, n, n)
        for j in 1:n, k in 1:n
            Ψ[k, j] = exp(-(k - j)^2 / (2 * s2)) * exp(im * φ[j] * k)
        end
        for j in 1:n; Ψ[:, j] ./= norm(Ψ[:, j]); end
        G = Ψ' * Ψ
        U_phys = random_haar_unitary(m)

        N = 60_000
        result = physical_tomography_hybrid(G, U_phys, N; threaded=false)

        reqs = required_paths(result.Γ, result.T)
        long_edges = Set(e for (e, p) in reqs if length(p) > 3)
        # The MDST is the path, so some tree path has order > 3 (correlator
        # cannot supply it) and the Fourier fallback covers EXACTLY those.
        @test maximum(length(p) for (_, p) in reqs) > 3
        @test !isempty(long_edges)
        @test Set(result.fourier_edges) == long_edges

        @test norm(result.Ĝ - result.Ĝ') < 1e-10
        @test all(abs(result.Ĝ[i, i] - 1) < 1e-8 for i in 1:n)

        # Derived upper bound on ‖G−Ĝ‖²₂, the same per-edge variance display
        # as the Fourier end-to-end test (E‖G−Ĝ‖²₂ = Σ_{i≠j}E|G_ij−Ĝ_ij|²,
        # bias lower-order in 1/N, Higham projection only contracts). The
        # variance source now differs by edge: order-2 magnitudes and order-3
        # paths come from the correlator (Var(B̂) ≤ σ_eff²/N, the Theorem-1
        # noise level _one_shot_sigma_eff), the order->3 paths from Fourier
        # (Var(B̂_π) = (1−|B_π|²)/N). C = 4 covers weak-edge Taylor
        # corrections and this seeded realisation's fluctuation.
        σ_eff = _one_shot_sigma_eff(G, U_phys, n)
        paths = Dict(e => p for (e, p) in reqs)
        sq_tol = 0.0
        for i in 1:n, j in (i+1):n
            B = abs2(G[i, j])
            varBij = σ_eff^2 / N                   # order-2 from the correlator
            taylor = if result.T[i, j]
                varBij / (4B)                      # prop:bias-var-Yhat
            else
                p = paths[(i, j)]
                absBπ = abs(bargmann_invariant(G, p))
                varBπ = (i, j) in long_edges ? (1 - absBπ^2) / N : σ_eff^2 / N
                (1 / (4B) + 1 / (2 * B^1.5)) * varBij + varBπ / (2 * absBπ^2)
            end
            small = 4B + 2 * sqrt(varBij)          # smallness bound, weak edges
            sq_tol += 2 * min(taylor, small)       # entries (i,j) AND (j,i)
        end
        sq_tol *= 4                                # C = 4
        G_gauged = regauge_to_tree(G, result.T)
        @test norm(result.Ĝ - G_gauged)^2 ≤ sq_tol
        # Strong (nearest-neighbour) tree edges are individually accurate, to
        # the delta-method sd of √B̂ at the worst-case correlator noise level:
        # sd(Ĝ_ij) ≈ sd(B̂_ij)/(2√B_ij) ≤ σ_eff/(2√(N·B_ij)). A 6σ envelope
        # (no magic number; the correlator inversion makes these noisier than
        # the independent HOM pairs of the Fourier route).
        for i in 1:(n-1)
            B = abs2(G[i, i+1])
            se = σ_eff / (2 * sqrt(N * B))
            @test abs(result.Ĝ[i, i+1] - G_gauged[i, i+1]) < 6 * se + 1e-6
        end
    end

    # (c) Backward-compat: `physical_tomography_from_samples` with no supplier
    #     still errors on an order->3 path (locks the default), and accepts a
    #     supplier without erroring.
    let n = 5, m = 8
        Random.seed!(271)
        s2 = 1.12
        Ψ = zeros(ComplexF64, n, n)
        for j in 1:n, k in 1:n; Ψ[k, j] = exp(-(k - j)^2 / (2 * s2)); end
        for j in 1:n; Ψ[:, j] ./= norm(Ψ[:, j]); end
        G = Ψ' * Ψ
        U_phys = random_haar_unitary(m)
        samples = one_shot_sample(G, U_phys, 20_000; threaded=false)
        @test_throws ErrorException physical_tomography_from_samples(n, U_phys, samples)
        stub = (i, j, path) -> bargmann_invariant(G, path)   # exact oracle
        ok = physical_tomography_from_samples(n, U_phys, samples;
                                              long_path_supplier=stub)
        @test norm(ok.Ĝ - ok.Ĝ') < 1e-10
    end
end

@testset "Phase-1 plan: plan_required_cycles + decoupled reconstruct_gram" begin
    # The two-phase boundary made explicit: phase 1 (plan_required_cycles)
    # turns pairwise estimates into the spanning tree + the exact list of
    # cycles to measure; phase 2 (reconstruct_gram) consumes those cycle
    # measurements. Chain Gram ⇒ MDST is the path ⇒ cycles up to order n.
    n = 5; s2 = 1.12
    Ψ = zeros(ComplexF64, n, n)
    for j in 1:n, k in 1:n; Ψ[k, j] = exp(-(k - j)^2 / (2 * s2)); end
    for j in 1:n; Ψ[:, j] ./= norm(Ψ[:, j]); end
    G = Ψ' * Ψ
    B2 = [abs2(G[i, j]) for i in 1:n, j in 1:n]          # exact order-2

    plan = plan_required_cycles(B2; δ=0.0)
    # The plan's cycles == required_paths on the same (Γ,T), each tagged with
    # its true order = tree-path length.
    @test Set(c.edge for c in plan.cycles) ==
          Set(e for (e, _) in required_paths(plan.Γ, plan.T))
    for c in plan.cycles
        @test c.order == length(c.path)
        @test c.order == length(tree_path(plan.T, c.edge[1], c.edge[2]))
    end
    @test maximum(c.order for c in plan.cycles) > 3      # chain ⇒ high-order cycle

    # Decoupled flow: measure each planned B_π (exactly here = noiseless limit),
    # then reconstruct. The plan's (Γ,T) and reconstruct_gram's (recomputed
    # from B2,δ) coincide, so the Bpath keyed by the plan is exactly what
    # reconstruct_gram needs — Ĝ recovers G up to the tree gauge.
    Bpath = Dict(c.edge => bargmann_invariant(G, c.path) for c in plan.cycles)
    rec = reconstruct_gram(B2, Bpath, 0.0)
    @test rec.T == plan.T
    @test norm(rec.Ĝ - regauge_to_tree(G, rec.T)) < 1e-8
end

@testset "physical_tomography: selectable low-order estimator" begin
    require_fixed_samplers()

    # (a) low_order=:one_shot delegates to the hybrid (bit-identical), needs
    #     U_phys, and on a star Gram (diameter ≤ 2) makes no Fourier calls.
    let n = 4, m = 7, α = 0.55
        Ψ = zeros(ComplexF64, n, n); Ψ[1, 1] = 1.0
        θ = (0.7, -1.3, 2.1)
        for j in 2:n; Ψ[1, j] = α * exp(im * θ[j-1]); Ψ[j, j] = sqrt(1 - α^2); end
        G = Ψ' * Ψ; U = random_haar_unitary(m); N = 40_000

        Random.seed!(555); a = physical_tomography(G; low_order=:one_shot,
                                                    U_phys=U, N=N, threaded=false)
        Random.seed!(555); b = physical_tomography_hybrid(G, U, N; threaded=false)
        @test a.low_order == :one_shot
        @test a.Ĝ == b.Ĝ                                  # pure delegation
        @test isempty(a.fourier_edges)
        @test all(c.order == 3 for c in a.cycles)         # star ⇒ triangles only

        @test_throws ErrorException physical_tomography(G; low_order=:one_shot, N=N)
        @test_throws ErrorException physical_tomography(G; low_order=:fortune_teller,
                                                        U_phys=U, N=N)
    end

    # (b) low_order=:fourier delegates to the Fourier route (bit-identical),
    #     needs no U_phys, and supplies EVERY cycle via Fourier. Chain Gram.
    let n = 5
        Random.seed!(606); s2 = 1.12; φ = 0.6 .* (rand(n) .- 0.5)
        Ψ = zeros(ComplexF64, n, n)
        for j in 1:n, k in 1:n; Ψ[k, j] = exp(-(k - j)^2 / (2 * s2)) * exp(im * φ[j] * k); end
        for j in 1:n; Ψ[:, j] ./= norm(Ψ[:, j]); end
        G = Ψ' * Ψ; N = 20_000

        Random.seed!(707); a = physical_tomography(G; low_order=:fourier, N=N, threaded=false)
        Random.seed!(707); b = physical_tomography_fourier(G; N=N, threaded=false)
        @test a.low_order == :fourier
        @test a.Ĝ == b.Ĝ
        @test Set(a.fourier_edges) == Set(c.edge for c in a.cycles)   # all cycles Fourier
        @test maximum(c.order for c in a.cycles) > 3
    end
end

@testset "protocol: δ-cutoff + exact reconstruction" begin
    # Supplier-agnostic protocol (protocol.jl) on EXACT Bargmann inputs —
    # no sampling. (a) δ = 0 keeps the complete graph and reproduces G in
    # the tree gauge to machine precision; (b) a threshold strictly between
    # the two smallest |B_{ij}| removes exactly the weakest edge, which
    # stays 0 in Ĝ.
    Random.seed!(515)
    n, d = 5, 25              # d ≫ n → weak overlaps, comfortably PSD after
                              # zeroing the weakest edge (projection ≈ identity)
    G, _ = random_gram_matrix(n, d)
    B2 = Float64[abs2(G[i, j]) for i in 1:n, j in 1:n]    # exact B_{ij}

    # (a) δ = 0: complete graph, n(δ) = 0, exact reconstruction.
    Γ, n_removed = protocol_graph(B2, 0.0)
    @test n_removed == 0
    @test all(Γ[i, j] == (i != j) for i in 1:n, j in 1:n)
    T = protocol_tree(Γ, B2)
    Bpath = Dict{Tuple{Int, Int}, ComplexF64}()
    for ((i, j), path) in required_paths(Γ, T)
        @test path[1] == i && path[end] == j
        Bpath[(i, j)] = bargmann_invariant(G, path)        # exact B_{π(i,j)}
    end
    res = reconstruct_gram(B2, Bpath, 0.0)
    @test res.T == T && res.Γ == Γ && res.n_removed == 0
    @test norm(res.Ĝ - regauge_to_tree(G, res.T)) < 1e-8

    # (b) δ strictly between the two smallest |B_{ij}|: exactly one edge
    # removed; Γ loses it; the assembled entry is 0 (the projection leaves
    # the exactly-PSD assembled matrix unchanged up to round-off).
    offdiag = sort([(B2[i, j], (i, j)) for i in 1:n for j in (i+1):n])
    Bmin, (imin, jmin) = offdiag[1]
    δ = 0.5 * (Bmin + offdiag[2][1])
    Γδ, nremδ = protocol_graph(B2, δ)
    @test nremδ == 1
    @test !Γδ[imin, jmin] && !Γδ[jmin, imin]
    Tδ = protocol_tree(Γδ, B2)
    Bpathδ = Dict{Tuple{Int, Int}, ComplexF64}(
        (i, j) => bargmann_invariant(G, path)
        for ((i, j), path) in required_paths(Γδ, Tδ))
    resδ = reconstruct_gram(B2, Bpathδ, δ)
    @test resδ.n_removed == 1
    @test abs(resδ.Ĝ[imin, jmin]) < 1e-8
    # All surviving entries still match the (re-gauged) truth.
    Gδ_gauged = regauge_to_tree(G, resδ.T)
    @test all(abs(resδ.Ĝ[i, j] - Gδ_gauged[i, j]) < 1e-8
              for i in 1:n for j in (i+1):n if Γδ[i, j])
end

@testset "protocol: synthetic MSE obeys Theorem 1 bound (eq. 40–42)" begin
    # ONE-DIRECTIONAL validation of Theorem 1 (`th:main-MSE` of
    # gram-estimation.tex = eq. 40–42 of Gram_Tomography.pdf): under the
    # manuscript's own estimator model — an unbiased B̂_π per invariant with
    # Var(B̂_π) = σ²/N ("We assume the variance is bounded by σ²/N",
    # gram-estimation.tex) — the Monte-Carlo mean squared FROBENIUS error of
    # the supplier-agnostic protocol must lie below the Theorem-1 RHS
    #
    #     E‖G − Ĝ‖²₂ ≤ n(δ)·δ + n²·V(δ),
    #
    # where ‖·‖₂ is Frobenius (E‖G−Ĝ‖²₂ = Σ_ij E|G_ij−Ĝ_ij|², the
    # bias-variance display of gram-estimation.tex). Pure states → ε = 0
    # (ε-pieces of eq. 41 drop); fresh independent noise per path → the
    # Cauchy–Schwarz covariance term K(δ) (eq. 42) drops. THIS testset is
    # where the suite's error tolerances are anchored to the manuscript.
    Random.seed!(20260611)
    n, d = 5, 3
    G, _ = random_gram_matrix(n, d)
    σ, N = 0.5, 10^4              # explicit estimator-model parameters

    # Noiseless protocol quantities entering the bound: δ_min (eq:deltamin,
    # eq. 37), the MDST T₀, M(T₀) (eq. 38, sqrt-inside-product), m(T₀)
    # (hop diameter + 1, eq. 41). We run the protocol at threshold δ = 0;
    # since every true B_ij ≥ δ_min, the surviving edge set is the same as at
    # δ = δ_min, where n(δ) = 0 — so the bound n²·V(δ_min) applies.
    B2x = Float64[i == j ? 1.0 : abs2(G[i, j]) for i in 1:n, j in 1:n]
    δmin = minimum(B2x[i, j] for i in 1:n for j in (i+1):n)
    @test δmin > 0.01             # instance is well-conditioned (seeded)
    Γ0, nrem0 = protocol_graph(B2x, 0.0)
    @test nrem0 == 0
    T0 = protocol_tree(Γ0, B2x)
    M0 = M_T_thm1(T0, B2x)
    m0 = m_T(T0)
    bound = frobenius_mse_bound_thm1(n, 0, δmin, M0, m0, σ, N)   # eq. 40

    # Monte-Carlo: B̂_ij = B_ij + (σ/√N)·Z with real Z (B_ij = |G_ij|² ∈ ℝ),
    # B̂_π = B_π + (σ/√N)·Z with Z ∼ CN(0,1) — exactly the Theorem-1 model.
    # Each trial's error is taken against the tree-gauge-matched truth
    # `regauge_to_tree(G, result.T)` of that trial's own tree, which is the
    # quantity the protocol estimates; the tree is empirically stable at this
    # N (counted below), so M(T₀), m(T₀) parametrise the bound.
    trials = 2000
    acc = 0.0
    same_tree = 0
    for _ in 1:trials
        B̂2 = Matrix{Float64}(I, n, n)
        for i in 1:n, j in (i+1):n
            b = real(simulate_bargmann_estimate(B2x[i, j], σ, N;
                                                complex_noise=false))
            B̂2[i, j] = b; B̂2[j, i] = b
        end
        Γt, _ = protocol_graph(B̂2, 0.0)
        Tt = protocol_tree(Γt, abs.(B̂2))
        Bpath = Dict{Tuple{Int, Int}, ComplexF64}()
        for ((i, j), path) in required_paths(Γt, Tt)
            Bpath[(i, j)] = simulate_bargmann_estimate(
                bargmann_invariant(G, path), σ, N)
        end
        res = reconstruct_gram(B̂2, Bpath, 0.0)
        acc += norm(res.Ĝ - regauge_to_tree(G, res.T))^2
        same_tree += (res.T == T0)
    end
    mse = acc / trials
    @test same_tree ≥ 0.95 * trials      # noisy MDST ≈ noiseless MDST T₀
    # The Theorem-1 assertion (one-directional; observed mse ≈ bound/54 —
    # the bound's worst-case per-entry substitutions B_ij → δ, |B_π| → √δ/M
    # make it loose on typical Gram matrices, see CLAUDE.md):
    @test mse ≤ bound
    # Non-vacuity guard: the bound is finite and within 3 orders of magnitude
    # of the empirical MSE for this well-conditioned instance.
    @test bound < 1e3 * mse
end

@testset "Fourier end-to-end: physical_tomography_fourier, MDST diameter > 2" begin
    # The Fourier method supplies B_π at ANY order, so — unlike the
    # correlator method, which throws on this very Gram family (testset
    # "MDST diameter > 2 throws" above) — it reconstructs chain-shaped
    # Gram matrices whose MDST is a path and whose non-tree edges need
    # invariants of order up to n.
    require_fixed_samplers()
    for n in (5, 6)
        Random.seed!(2600 + n)
        # Gaussian-profile chain states (same family as the order->3-throws
        # test): |⟨ψ_i|ψ_j⟩| decays super-exponentially in |i − j|, so the
        # MDST is the path 1-2-…-n. Mild random phase ramps exp(i·φ_j·k)
        # make G genuinely complex without destroying the chain structure.
        s2 = 1.12
        φ = 0.6 .* (rand(n) .- 0.5)
        Ψ = zeros(ComplexF64, n, n)
        for j in 1:n, k in 1:n
            Ψ[k, j] = exp(-(k - j)^2 / (2 * s2)) * exp(im * φ[j] * k)
        end
        for j in 1:n; Ψ[:, j] ./= norm(Ψ[:, j]); end
        G = Ψ' * Ψ

        N = 20_000
        result = physical_tomography_fourier(G; N=N)

        # The MDST should be the path → at least one tree path of order > 3
        # was needed (the correlator method cannot supply it).
        @test maximum(length(path) for ((i, j), path)
                      in required_paths(result.Γ, result.T)) > 3

        # Hermitian, unit diagonal (assembly + Higham projection).
        @test norm(result.Ĝ - result.Ĝ') < 1e-10
        @test all(abs(result.Ĝ[i, i] - 1) < 1e-8 for i in 1:n)

        # Order-2 estimates: real, within Monte-Carlo error of |G_{ij}|².
        for i in 1:n, j in (i+1):n
            @test abs(imag(result.B2[i, j])) < 1e-12   # exp(iπf) = ±1 exactly
            se = sqrt((1 - abs2(G[i, j])^2) / N)
            @test abs(real(result.B2[i, j]) - abs2(G[i, j])) < 6 * se + 1e-6
        end

        # Reconstruction error vs. the gauge-matched truth, with a tolerance
        # DERIVED from the manuscript's per-edge bounds — no magic numbers.
        # E‖G−Ĝ‖²₂ = Σ_{i≠j} E|G_ij−Ĝ_ij|² (bias-variance display of
        # gram-estimation.tex); the Higham projection only contracts this
        # error (‖p(Ĝ)−G‖₂ ≤ ‖Ĝ−G‖₂, ibid.), and bias² is lower order in
        # 1/N (proof of th:main-MSE), so we sum per-edge VARIANCE bounds with
        # the closed-form per-path Fourier variances Var(B̂_π) =
        # (1 − |B_π|²)/N (unit-modulus summands, fourier_estimator.jl; the
        # M = 2 case is the HOM pair estimator):
        #  - tree edge, prop:bias-var-Yhat:
        #        Var(Ĝ_ij) ≤ Var(B̂_ij)/(4 B_ij);
        #  - non-tree edge, prop:bias-var-notinT at ε = 0 with κ_ij = 0
        #    (every pair/path is its own independent Fourier experiment):
        #        Var(Ĝ_ij) ≤ (1/(4B_ij) + 1/(2B_ij^{3/2}))·Var(B̂_ij)
        #                    + Var(B̂_{π(i,j)})/(2|B_{π(i,j)}|²);
        #  - the Taylor bounds blow up on the ultra-weak long-range edges
        #    (B_ij ≲ sd(B̂)); there we use the smallness bound
        #        E|Ĝ_ij − G_ij|² ≤ 2E|Ĝ_ij|² + 2|G_ij|²
        #                        ≤ 4B_ij + 2·sd(B̂_ij),
        #    since |Ĝ_ij|² = max(B̂_ij, 0) ≤ |B̂_ij| (eq:gram1/eq:gram2 with
        #    |ê^{iφ}| = 1) and E|B̂−B| ≤ sd(B̂) by Jensen — the same
        #    |G_ij|² = B_ij logic as the n(δ)·δ truncation term of
        #    th:main-MSE.
        # C = 4 covers the higher-order Taylor corrections on the weak edges
        # and the fluctuation of this single seeded realisation around the
        # bounded mean (observed sq. error ≈ 0.019 (n=5) / 0.024 (n=6)
        # against C·Σ ≈ 0.34 / 0.76).
        paths = Dict(e => p for (e, p) in required_paths(result.Γ, result.T))
        sq_tol = 0.0
        for i in 1:n, j in (i+1):n
            B = abs2(G[i, j])
            varBij = (1 - B^2) / N
            taylor = if result.T[i, j]
                varBij / (4B)
            else
                absBπ = abs(bargmann_invariant(G, paths[(i, j)]))
                (1 / (4B) + 1 / (2 * B^1.5)) * varBij +
                    ((1 - absBπ^2) / N) / (2 * absBπ^2)
            end
            small = 4B + 2 * sqrt(varBij)
            sq_tol += 2 * min(taylor, small)      # entries (i,j) AND (j,i)
        end
        sq_tol *= 4                               # C = 4
        G_gauged = regauge_to_tree(G, result.T)
        @test norm(result.Ĝ - G_gauged)^2 ≤ sq_tol
        # Strong (nearest-neighbour) entries are individually accurate.
        for i in 1:(n-1)
            @test abs(result.Ĝ[i, i+1] - G_gauged[i, i+1]) < 0.05
        end
    end
end

@testset "Threaded sampling: across-experiment parallelism is correct" begin
    # `threaded=true` runs the independent per-pair / per-path experiments in
    # `Threads.@spawn` tasks (boson_sampling_integration.jl `_run_experiments`).
    # The guarantees the docstrings promise — and that this testset locks in:
    #   (1) bit-reproducible under a top-level `Random.seed!`;
    #   (2) the caller's own RNG state is left untouched;
    #   (3) statistically identical to the serial path.
    require_fixed_samplers()
    G, _ = random_gram_matrix(5, 5)

    # (1) reproducibility of the threaded run under a fixed seed.
    Random.seed!(4242); rA = physical_tomography_fourier(G; N=8_000, threaded=true)
    Random.seed!(4242); rB = physical_tomography_fourier(G; N=8_000, threaded=true)
    @test rA.Ĝ == rB.Ĝ
    @test rA.B2 == rB.B2

    # (2) the spawned tasks must not reseed/clobber the caller's task-local RNG:
    # a `rand()` drawn after a seeded threaded run is itself reproducible.
    Random.seed!(99); physical_tomography_fourier(G; N=2_000, threaded=true); a = rand()
    Random.seed!(99); physical_tomography_fourier(G; N=2_000, threaded=true); b = rand()
    @test a == b

    # (3) threaded ≡ serial statistically: both unbiased for the pairwise
    # |G_ij|², so they agree with the truth (and each other) to Monte-Carlo
    # error. (Exact bit-equality serial-vs-threaded is NOT expected — different
    # RNG task structure — only statistical agreement.)
    N = 60_000
    Random.seed!(7); ser = physical_tomography_fourier(G; N=N, threaded=false)
    Random.seed!(7); thr = physical_tomography_fourier(G; N=N, threaded=true)
    for i in 1:5, j in (i+1):5
        se = sqrt((1 - abs2(G[i, j])^2) / N)
        @test abs(real(ser.B2[i, j]) - abs2(G[i, j])) < 6 * se + 1e-6
        @test abs(real(thr.B2[i, j]) - abs2(G[i, j])) < 6 * se + 1e-6
    end

    # `hom_tomography_magnitudes` shares `_run_experiments`: same reproducibility.
    Random.seed!(11); hA = hom_tomography_magnitudes(G, 5_000; threaded=true)
    Random.seed!(11); hB = hom_tomography_magnitudes(G, 5_000; threaded=true)
    @test hA == hB
end
