using Gmsh
using Gridap, GridapGmsh
using Gridap.CellData
using Gridap.Geometry
using Gridap.FESpaces
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random
using CairoMakie

# =============================================================================
# 1. Constants
# =============================================================================
const σ_default = 0.25
const σ_art     = 0.05            # artificial viscosity for σ=0 deterministic case
const κ         = 0.1             # congestion strength
const ε_log     = 1e-3            # log regularization
const T_final   = 1.0
const Nt_steps  = 40
const Picard_K  = 15
const Picard_θ  = 0.3
const Picard_tol= 1e-3
const m0_center = (0.20, 0.50)
const m0_width  = 0.10
const exit_y    = (0.45, 0.55)
const default_h = 0.04

@inline safe_log(x) = log(max(x, 1e-12))

# =============================================================================
# 2. Mesh
# =============================================================================
"""
    build_default_mesh(path; h, exit_y)

Generate a unit-square room with a small exit on the right wall.
Boundary tagged "wall" (no-flux) and "exit" (Dirichlet). Returns the file path.
"""
function build_default_mesh(path::String; h::Float64=default_h,
                            exit_y::Tuple{Float64,Float64}=exit_y)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("room")
    g = gmsh.model.geo
    p1 = g.addPoint(0.0, 0.0, 0.0, h)
    p2 = g.addPoint(1.0, 0.0, 0.0, h)
    p3 = g.addPoint(1.0, exit_y[1], 0.0, h)
    p4 = g.addPoint(1.0, exit_y[2], 0.0, h)
    p5 = g.addPoint(1.0, 1.0, 0.0, h)
    p6 = g.addPoint(0.0, 1.0, 0.0, h)
    l1 = g.addLine(p1, p2); l2 = g.addLine(p2, p3); l3 = g.addLine(p3, p4)
    l4 = g.addLine(p4, p5); l5 = g.addLine(p5, p6); l6 = g.addLine(p6, p1)
    cl = g.addCurveLoop([l1,l2,l3,l4,l5,l6])
    s  = g.addPlaneSurface([cl])
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1,l2,l4,l5,l6], -1, "wall")
    gmsh.model.addPhysicalGroup(1, [l3],              -1, "exit")
    gmsh.model.addPhysicalGroup(2, [s],               -1, "domain")
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

"""
    load_mesh(path)

Load a `.msh` file and return a `GmshDiscreteModel`. The file must tag
boundary segments with the physical names "wall" and "exit".
"""
load_mesh(path::String) = GmshDiscreteModel(path)

# =============================================================================
# 3. FE setup
# =============================================================================
struct MFGSpaces
    model
    Ω
    dΩ        :: Measure
    Γe
    dΓe       :: Measure
    n_Γe
    V_u; U_u
    V_m; U_m
end

function build_spaces(model)
    reffe = ReferenceFE(lagrangian, Float64, 1)
    V_u = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["exit"])
    U_u = TrialFESpace(V_u, 0.0)
    V_m = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["exit"])
    U_m = TrialFESpace(V_m, 0.0)
    Ω  = Triangulation(model);          dΩ  = Measure(Ω, 2)
    Γe = BoundaryTriangulation(model; tags="exit"); dΓe = Measure(Γe, 2)
    return MFGSpaces(model, Ω, dΩ, Γe, dΓe, get_normal_vector(Γe),
                     V_u, U_u, V_m, U_m)
end

function init_density(S::MFGSpaces)
    bump(x) = exp(-((x[1]-m0_center[1])^2 + (x[2]-m0_center[2])^2)/(2*m0_width^2))
    raw = interpolate_everywhere(bump, S.U_m)
    M = sum(∫(raw)*S.dΩ)
    interpolate_everywhere(x -> bump(x)/M, S.U_m)
end

# =============================================================================
# 4. Single-step solvers
# =============================================================================
"""
    solve_hjb_step(S, u_next, m_n, u_prev_n, σ, dt) -> u^n

Implicit-Euler step of the HJB *backwards* in time: given the future value
`u_next = u^{n+1}`, the current density `m_n`, and the previous Picard iterate
`u_prev_n` whose gradient lags the nonlinearity, solve

    u^n − (σ²Δt/2) Δu^n + (Δt/2) ∇u_prev · ∇u^n = u^{n+1} + Δt · κ log(m^n + ε)

with u^n = 0 on Γ_e and zero-Neumann on Γ_w.
"""
function solve_hjb_step(S::MFGSpaces, u_next::FEFunction, m_n::FEFunction,
                        u_prev_n::FEFunction, σ::Float64, dt::Float64)
    F_m = κ * (safe_log∘(m_n + ε_log))
    a(u, v) = ∫( u*v +
                 (σ^2*dt/2) * (∇(u) ⋅ ∇(v)) +
                 (dt/2)     * ((∇(u_prev_n) ⋅ ∇(u)) * v) ) * S.dΩ
    l(v)   = ∫( u_next*v + dt * F_m * v ) * S.dΩ
    op = AffineFEOperator(a, l, S.U_u, S.V_u)
    return solve(op)
end

"""
    solve_fp_step(S, m_prev, u_np1, σ, dt) -> m^{n+1}

Implicit-Euler forward step of the Fokker–Planck equation:

    m^{n+1} − (σ²Δt/2) Δm^{n+1} − Δt ∇·(m^{n+1} ∇u^{n+1}) = m^n

Boundary: m=0 on Γ_e (absorbing), no-flux on Γ_w.
"""
function solve_fp_step(S::MFGSpaces, m_prev::FEFunction, u_np1::FEFunction,
                       σ::Float64, dt::Float64)
    a(m, v) = ∫( m*v +
                 (σ^2*dt/2) * (∇(m) ⋅ ∇(v)) +
                 dt * (m * (∇(u_np1) ⋅ ∇(v))) ) * S.dΩ
    l(v)    = ∫( m_prev * v ) * S.dΩ
    op = AffineFEOperator(a, l, S.U_m, S.V_m)
    return solve(op)
end

# =============================================================================
# 5. Picard loop
# =============================================================================
struct MFGSolution
    u_traj :: Vector{FEFunction}
    m_traj :: Vector{FEFunction}
    times  :: Vector{Float64}
    res    :: Vector{Float64}
    cost   :: Vector{Float64}
    σ_eff  :: Float64
    spaces :: MFGSpaces
end

function l2_norm_dt(S, traj_a, traj_b, dt)
    r2 = 0.0
    for n in 1:length(traj_a)
        d = traj_a[n] - traj_b[n]
        r2 += sum(∫(d*d)*S.dΩ) * dt
    end
    return sqrt(r2)
end

"""
    picard_loop(S; σ, T, Nt, K_max, tol, θ, stochastic) -> MFGSolution

Lagged-Picard fixed-point iteration. At each iterate:
  1. solve HJB backward with ∇u^{(k-1)} lagged in the |∇u|² term;
  2. solve FP forward using the new u^(k);
  3. damped update m^(k) = θ·m̃ + (1−θ)·m^(k−1) (and similarly for u_prev).

Set `stochastic=false` to switch to σ=0 (deterministic agents); a small
artificial viscosity σ_art is then added to keep the central-Galerkin scheme
stable.
"""
function picard_loop(S::MFGSpaces; σ::Float64=σ_default, T::Float64=T_final,
                     Nt::Int=Nt_steps, K_max::Int=Picard_K, tol::Float64=Picard_tol,
                     θ::Float64=Picard_θ, stochastic::Bool=true,
                     verbose::Bool=true)
    σ_eff = stochastic ? σ : σ_art
    dt = T/Nt
    times = collect(0:Nt) .* dt

    m0 = init_density(S)
    m_traj = [FEFunction(S.U_m, copy(get_free_dof_values(m0))) for _ in 1:(Nt+1)]
    u_prev = [FEFunction(S.U_u, zeros(num_free_dofs(S.U_u))) for _ in 1:(Nt+1)]
    uT     = FEFunction(S.U_u, zeros(num_free_dofs(S.U_u)))   # g(x)=0

    res_hist  = Float64[]
    cost_hist = Float64[]
    u_traj    = u_prev   # placeholder

    for k in 1:K_max
        # backward HJB sweep
        u_traj = Vector{FEFunction}(undef, Nt+1)
        u_traj[Nt+1] = uT
        for n in Nt:-1:1
            u_traj[n] = solve_hjb_step(S, u_traj[n+1], m_traj[n], u_prev[n], σ_eff, dt)
        end

        # forward FP sweep
        m_tilde = Vector{FEFunction}(undef, Nt+1)
        m_tilde[1] = m_traj[1]
        for n in 1:Nt
            m_tilde[n+1] = solve_fp_step(S, m_tilde[n], u_traj[n+1], σ_eff, dt)
        end

        # residual on m
        push!(res_hist, l2_norm_dt(S, m_tilde, m_traj, dt))

        # damped updates on m and u_prev
        m_new = Vector{FEFunction}(undef, Nt+1)
        for n in 1:(Nt+1)
            v = θ*get_free_dof_values(m_tilde[n]) + (1-θ)*get_free_dof_values(m_traj[n])
            m_new[n] = FEFunction(S.U_m, v)
        end
        m_traj = m_new

        u_new = Vector{FEFunction}(undef, Nt+1)
        for n in 1:(Nt+1)
            v = θ*get_free_dof_values(u_traj[n]) + (1-θ)*get_free_dof_values(u_prev[n])
            u_new[n] = FEFunction(S.U_u, v)
        end
        u_prev = u_new

        # cost = ∫₀ᵀ ∫_Ω (½|∇u|² + κ log(m+ε)) m dx dt   (trapezoidal in t)
        c = 0.0
        for n in 1:(Nt+1)
            wt = (n==1 || n==Nt+1) ? 0.5*dt : dt
            integrand = (0.5*(∇(u_traj[n])⋅∇(u_traj[n])) +
                         κ*(safe_log∘(m_traj[n]+ε_log))) * m_traj[n]
            c += sum(∫(integrand)*S.dΩ) * wt
        end
        push!(cost_hist, c)

        verbose && @printf "  Picard k=%2d:  ‖Δm‖ = %.3e   cost = %.3e\n" k res_hist[end] cost_hist[end]
        if res_hist[end] < tol
            verbose && println("  converged.")
            break
        end
    end
    MFGSolution(u_traj, m_traj, times, res_hist, cost_hist, σ_eff, S)
end

# =============================================================================
# 6. Agent simulation
# =============================================================================
"""
    sample_initial(S, m0, n) -> Vector{Point2}

Rejection-sample `n` initial agent positions from the density `m0`.
"""
function sample_initial(S::MFGSpaces, m0::FEFunction, n::Int;
                        bbox=(0.0,1.0,0.0,1.0), rng=Random.default_rng())
    pts = Vector{Tuple{Float64,Float64}}(undef, n)
    xmin, xmax, ymin, ymax = bbox
    Mmax = 1.05 * maximum(get_free_dof_values(m0))   # rough sup
    i = 1
    while i ≤ n
        x = xmin + (xmax-xmin)*rand(rng); y = ymin + (ymax-ymin)*rand(rng)
        local val
        try
            val = m0(Point(x,y))
        catch
            continue   # outside domain
        end
        if val > 0 && rand(rng)*Mmax < val
            pts[i] = (x, y); i += 1
        end
    end
    return pts
end

"""
    simulate_agents(sol; n_agents, stochastic, rng) -> Vector of trajectories

Euler–Maruyama for closed-loop SDE dX = -∇u(t,X) dt + σ dW, with absorbing
boundary on Γ_e. Returns one trajectory per agent: a vector of (t, x, y, alive).
"""
function simulate_agents(sol::MFGSolution; n_agents::Int=200,
                         stochastic::Bool=true, rng=Random.default_rng())
    S = sol.spaces
    times = sol.times
    Nt = length(times)-1
    dt = times[2] - times[1]
    σ = stochastic ? sol.σ_eff : 0.0

    init_pts = sample_initial(S, sol.m_traj[1], n_agents; rng=rng)
    # trajectory[i] = matrix (Nt+1, 2), with NaN after exit
    traj = [fill(NaN, Nt+1, 2) for _ in 1:n_agents]
    alive = trues(n_agents)
    for i in 1:n_agents
        traj[i][1, 1] = init_pts[i][1]
        traj[i][1, 2] = init_pts[i][2]
    end

    for n in 1:Nt
        u_h = sol.u_traj[n]
        for i in 1:n_agents
            alive[i] || continue
            x, y = traj[i][n, 1], traj[i][n, 2]
            local g
            try
                g = (∇(u_h))(Point(x, y))
            catch
                alive[i] = false; continue
            end
            dx = -g[1]*dt; dy = -g[2]*dt
            if stochastic
                dx += σ * sqrt(dt) * randn(rng)
                dy += σ * sqrt(dt) * randn(rng)
            end
            xn, yn = x+dx, y+dy
            # Reflect off the box walls (kept simple for unit-square default mesh)
            xn = clamp(xn, 1e-6, 1-1e-6)
            yn = clamp(yn, 1e-6, 1-1e-6)
            # Exit detection
            if xn ≥ 1 - 2e-6 && exit_y[1] ≤ yn ≤ exit_y[2]
                alive[i] = false
            else
                traj[i][n+1, 1] = xn
                traj[i][n+1, 2] = yn
            end
        end
    end
    return traj
end

# =============================================================================
# 7. Visualization helpers (regular-grid sampling)
# =============================================================================
"""
    sample_field_grid(uh, nx, ny; bbox) -> (xs, ys, U)

Sample an FE function on a regular grid, returning NaN outside the domain.
"""
function sample_field_grid(uh, nx::Int=80, ny::Int=80; bbox=(0.0,1.0,0.0,1.0))
    xs = range(bbox[1], bbox[2]; length=nx)
    ys = range(bbox[3], bbox[4]; length=ny)
    U  = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        try
            U[i,j] = uh(Point(xs[i], ys[j]))
        catch
            # outside domain: leave NaN
        end
    end
    return collect(xs), collect(ys), U
end

# =============================================================================
# 8. Default scenario solve (smoke test)
# =============================================================================
function run_smoke_test()
    msh = joinpath(@__DIR__, "room.msh")
    build_default_mesh(msh; h=default_h)
    model = load_mesh(msh)
    S = build_spaces(model)
    sol = picard_loop(S)
    @printf "Final residual: %.3e\n" sol.res[end]
    @printf "Mass at t=0:   %.5f\n" sum(∫(sol.m_traj[1])*S.dΩ)
    @printf "Mass at t=T:   %.5f\n" sum(∫(sol.m_traj[end])*S.dΩ)

    # Quick visualization sanity test
    xs, ys, M = sample_field_grid(sol.m_traj[end], 60, 60)
    fig = Figure(; size=(400, 400))
    ax = Axis(fig[1,1]; aspect=1, title="m(T,x)")
    heatmap!(ax, xs, ys, M; colormap=:viridis)
    save(joinpath(@__DIR__, "smoke_density.png"), fig)
    println("Saved smoke_density.png")

    # Agent sim
    @time traj = simulate_agents(sol; n_agents=50)
    println("Simulated ", length(traj), " trajectories")
    return sol, traj
end

@time sol, traj = run_smoke_test()
println("ok")
