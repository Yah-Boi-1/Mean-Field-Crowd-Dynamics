#=
Builds mfg_crowd_dynamics.ipynb from a list of (type, content) cells defined
inline below.  Run with: julia --project=. scratch/build_notebook.jl
=#

using JSON
using UUIDs

const NOTEBOOK_PATH = joinpath(@__DIR__, "..", "mfg_crowd_dynamics.ipynb")

# Each cell: (kind=:md or :code, source::String).
const CELLS = Tuple{Symbol,String}[]

# helpers
addmd!(s)   = push!(CELLS, (:md, s))
addcode!(s) = push!(CELLS, (:code, s))

# =============================================================================
# 1. Setup & imports
# =============================================================================
addmd!(raw"""
# Mean Field Games for Crowd Evacuation — a 2D FEM tutorial in Julia

This notebook implements, solves, and visualizes a **second-order Mean Field Game (MFG)** for crowd evacuation on an unstructured triangular mesh, using [`Gridap.jl`](https://github.com/gridap/Gridap.jl) for FEM, [`GridapGmsh.jl`](https://github.com/gridap/GridapGmsh.jl)/[`Gmsh.jl`](https://github.com/JuliaFEM/Gmsh.jl) for meshing, and [`CairoMakie.jl`](https://docs.makie.org) for visualization. Each solver step is preceded by a markdown derivation. Treat the notebook as a graduate-level lecture handout.

The scenario is a square room with a narrow exit on the right wall. A density of agents `m(t,x)` evacuates while their cost-to-go `u(t,x)` is computed self-consistently, coupling a backward Hamilton–Jacobi–Bellman equation with a forward Fokker–Planck equation through a Picard fixed-point iteration.

## 1. Setup and imports

We activate the local project so that the package versions in `Project.toml`/`Manifest.toml` are used. First-run package precompilation can take a few minutes.
""")

addcode!(raw"""
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Gmsh
using Gridap, GridapGmsh
using Gridap.CellData, Gridap.Geometry, Gridap.FESpaces
using LinearAlgebra, SparseArrays, Statistics, Printf, Random
using CairoMakie

# Outputs directory (animations and figures land here)
const OUTPUTS = joinpath(@__DIR__, "outputs")
isdir(OUTPUTS) || mkdir(OUTPUTS)
println("Setup complete.")
""")

# =============================================================================
# 2. The MFG system
# =============================================================================
addmd!(raw"""
## 2. The MFG system

We work on a 2D domain $\Omega \subset \mathbb{R}^2$ whose boundary splits into walls $\Gamma_w$ (no-flux) and a small exit segment $\Gamma_e$ (Dirichlet). Each agent's state $X_t \in \Omega$ obeys the closed-loop SDE
$$dX_t = \alpha(t,X_t)\,dt + \sigma\,dB_t,$$
and incurs running cost $L(\alpha,m) = \tfrac12|\alpha|^2 + F(x,m)$, with absorbing exit on $\Gamma_e$. The mean-field interaction enters through the running cost via the **logarithmic congestion** term
$$F(x,m) = \kappa\,\log(m+\varepsilon),\qquad \kappa>0,\; \varepsilon \ll 1.$$

### 2.1 Hamilton–Jacobi–Bellman (HJB) — derivation

Let $u(t,x) := \inf_\alpha \mathbb{E}\!\left[\int_t^T L(\alpha_s, m_s)\,ds + g(X_T,m(T,X_T))\,\Big|\, X_t=x\right]$.

Dynamic programming gives, for a small $\delta t$,
$$u(t,x) = \inf_\alpha \mathbb{E}\!\left[L(\alpha,m)\delta t + u(t+\delta t, X_{t+\delta t})\right] + o(\delta t).$$
By Itô,
$$u(t+\delta t, X_{t+\delta t}) \approx u(t,x) + \big(\partial_t u + \alpha\!\cdot\!\nabla u + \tfrac{\sigma^2}{2}\Delta u\big)\delta t.$$
Substituting and dividing by $\delta t$,
$$0 = \inf_\alpha\!\left[\tfrac12|\alpha|^2 + F + \partial_t u + \alpha\!\cdot\!\nabla u + \tfrac{\sigma^2}{2}\Delta u\right].$$
The minimum in $\alpha$ is attained at the **optimal feedback** $\alpha^*(t,x) = -\nabla u(t,x)$, yielding
$$\boxed{\,-\partial_t u - \tfrac{\sigma^2}{2}\Delta u + \tfrac{1}{2}|\nabla u|^2 = F(x,m),\qquad u(T,x)=g(x,m(T,x))\,}.$$
The Hamiltonian is $H(x,p,m)=\tfrac12|p|^2 - F(x,m)$, the Legendre transform of the running cost in $\alpha$.

### 2.2 Fokker–Planck (FP)

With $\alpha=\alpha^*=-\nabla u$, the closed-loop SDE is $dX_t = -\nabla u(t,X_t)\,dt + \sigma\,dB_t$. The density $m=\rho_{X_t}$ satisfies the FP equation
$$\boxed{\,\partial_t m - \tfrac{\sigma^2}{2}\Delta m - \nabla\!\cdot\!\big(m\,\nabla u\big) = 0,\qquad m(0,x)=m_0(x)\,}.$$

### 2.3 Boundary conditions

* On the **walls** $\Gamma_w$: $\partial_\nu u = 0$ (no flux for cost) and $\partial_\nu m + \tfrac{2}{\sigma^2}\,m\,\partial_\nu u = 0$ (no probability flux; vanishes since $\partial_\nu u=0$, so it reduces to $\partial_\nu m = 0$).
* On the **exit** $\Gamma_e$: $u=0$ (terminal cost vanishes once you exit) and $m=0$ (absorbing).

### 2.4 Picard fixed point

For fixed $m$ the HJB is solved backward; for fixed $u$ the FP is solved forward. The MFG is the **fixed point** of the map
$$m \;\longmapsto\; u(m) \;\longmapsto\; m'(u),$$
which we approximate by Picard iteration. Existence/uniqueness in this loop is delicate when the cost $F(\cdot,m)$ is not monotone in $m$; logarithmic congestion is **not** Lasry–Lions monotone, so we treat the iteration carefully (damping; see §5).
""")

# =============================================================================
# 3. Domain, mesh, boundary tagging
# =============================================================================
addmd!(raw"""
## 3. Domain, mesh, and boundary tagging

We take $\Omega = (0,1)^2$ with a small exit on the right wall, $\Gamma_e = \{1\}\times[0.45,0.55]$, and the rest of $\partial\Omega$ as walls. We script the geometry in Gmsh and tag the boundary curves with **physical groups** named `"wall"` and `"exit"`. `GridapGmsh.GmshDiscreteModel` reads these tags into a face-labeling that we use to impose Dirichlet conditions and to integrate over $\Gamma_e$.

`build_default_mesh` regenerates the room each notebook run; `load_mesh` accepts any user-supplied `.msh` provided the same physical names are tagged.
""")

addcode!(raw"""
const exit_y_def    = (0.45, 0.55)
const default_h     = 0.04

\"\"\"
    build_default_mesh(path; h, exit_y) -> path

Generate a unit-square room with a narrow exit on the right wall and write
it to `path`.  Boundary segments tagged "wall" (no-flux) and "exit" (Dirichlet).
\"\"\"
function build_default_mesh(path::String; h::Float64=default_h,
                            exit_y::Tuple{Float64,Float64}=exit_y_def)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("room")
    g = gmsh.model.geo
    p1 = g.addPoint(0.0, 0.0,        0.0, h)
    p2 = g.addPoint(1.0, 0.0,        0.0, h)
    p3 = g.addPoint(1.0, exit_y[1],  0.0, h)
    p4 = g.addPoint(1.0, exit_y[2],  0.0, h)
    p5 = g.addPoint(1.0, 1.0,        0.0, h)
    p6 = g.addPoint(0.0, 1.0,        0.0, h)
    l1 = g.addLine(p1, p2); l2 = g.addLine(p2, p3); l3 = g.addLine(p3, p4)
    l4 = g.addLine(p4, p5); l5 = g.addLine(p5, p6); l6 = g.addLine(p6, p1)
    cl = g.addCurveLoop([l1,l2,l3,l4,l5,l6]); s = g.addPlaneSurface([cl])
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1,l2,l4,l5,l6], -1, "wall")
    gmsh.model.addPhysicalGroup(1, [l3],              -1, "exit")
    gmsh.model.addPhysicalGroup(2, [s],               -1, "domain")
    gmsh.model.mesh.generate(2)
    gmsh.write(path); gmsh.finalize()
    return path
end

\"\"\"
    load_mesh(path) -> GmshDiscreteModel

Load a `.msh` file. The file must use the physical names "wall" and "exit"
on its boundary segments.
\"\"\"
load_mesh(path::String) = GmshDiscreteModel(path)

# Generate the default mesh
mesh_path = joinpath(OUTPUTS, "room.msh")
build_default_mesh(mesh_path; h=default_h)
model = load_mesh(mesh_path)
labels = get_face_labeling(model)
println("tag → name : ", labels.tag_to_name)
println("# cells   : ", num_cells(model))
""")

addmd!(raw"""
A quick look at the unstructured triangulation: this is a Gridap `DiscreteModel` where each cell is a triangle and the boundary tags `wall`/`exit` are stored as face labels.
""")

addcode!(raw"""
let
    Ω = Triangulation(model)
    coords = collect(Gridap.Geometry.get_node_coordinates(Ω))
    cells  = collect(Gridap.Geometry.get_cell_node_ids(Ω))
    fig = Figure(; size=(450, 450))
    ax  = Axis(fig[1,1]; aspect=DataAspect(), title="Default mesh — $(num_cells(model)) triangles",
               xlabel="x", ylabel="y")
    for tri in cells
        idx = collect(tri); push!(idx, idx[1])
        xs = [coords[i][1] for i in idx]; ys = [coords[i][2] for i in idx]
        lines!(ax, xs, ys; color=(:gray, 0.4), linewidth=0.5)
    end
    # exit
    lines!(ax, [1.0, 1.0], [exit_y_def...]; color=:red, linewidth=4, label="exit")
    axislegend(ax; position=:lt)
    save(joinpath(OUTPUTS, "mesh.png"), fig)
    fig
end
""")

# =============================================================================
# 4. Weak formulations
# =============================================================================
addmd!(raw"""
## 4. Weak formulations (continuous P1 Lagrange)

We discretize both $u$ and $m$ in the same continuous P1 Lagrange space on the triangulation, with Dirichlet trace zero on $\Gamma_e$. Test and trial spaces share these BCs.

### 4.1 HJB weak form

Multiply the HJB by a test function $v\in H^1_0(\Omega; \Gamma_e)$ and integrate by parts the Laplacian (the boundary integral $\int_{\partial\Omega} \partial_\nu u\,v$ vanishes — on $\Gamma_w$ because $\partial_\nu u=0$, on $\Gamma_e$ because $v=0$). The $|\nabla u|^2$ term has no derivative on $u$ to integrate by parts, so it stays as a pointwise nonlinearity:
$$\int_\Omega -\partial_t u\,v + \tfrac{\sigma^2}{2}\nabla u\!\cdot\!\nabla v + \tfrac12 |\nabla u|^2\,v\,dx = \int_\Omega F(x,m)\,v\,dx.$$

### 4.2 FP weak form

Multiply the FP equation by $v\in H^1_0(\Omega;\Gamma_e)$ and integrate by parts both second-order operators. The flux is $J = -\tfrac{\sigma^2}{2}\nabla m + m\,(-\nabla u)$. Boundary integrals: on $\Gamma_w$ the BC enforces $J\!\cdot\!\nu=0$; on $\Gamma_e$ the test function vanishes. Thus
$$\int_\Omega \partial_t m\,v + \tfrac{\sigma^2}{2}\nabla m\!\cdot\!\nabla v + m\,\nabla u\!\cdot\!\nabla v\,dx = 0.$$

These are the two weak forms we will discretize in time and feed to Gridap's `AffineFEOperator`.
""")

addcode!(raw"""
\"\"\"
    MFGSpaces

Bundles the discrete model, the bulk and exit measures, and the matched
test/trial spaces for u and m (P1 Lagrange, Dirichlet on "exit").
\"\"\"
struct MFGSpaces
    model
    Ω; dΩ::Measure
    Γe; dΓe::Measure
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
    Ω  = Triangulation(model);                       dΩ  = Measure(Ω,  2)
    Γe = BoundaryTriangulation(model; tags="exit");  dΓe = Measure(Γe, 2)
    return MFGSpaces(model, Ω, dΩ, Γe, dΓe, get_normal_vector(Γe),
                     V_u, U_u, V_m, U_m)
end

S = build_spaces(model)
@printf "FE DOFs:  u_free=%d   m_free=%d\n" num_free_dofs(S.U_u) num_free_dofs(S.U_m)
""")

# =============================================================================
# 5. Time discretization & lagged-Picard scheme
# =============================================================================
addmd!(raw"""
## 5. Time discretization and the lagged-Picard scheme

We use a uniform time grid $0=t_0<\dots<t_N=T$ with $\Delta t = T/N$. The HJB is **backward** in time (terminal $u(T)$ is given), the FP **forward** in time (initial $m_0$ is given). We use **implicit Euler** for both, which is unconditionally stable for the parabolic part.

### 5.1 HJB step (backward Euler with lagged $|\nabla u|^2$)

At time level $n$, given $u^{n+1}$ and $m^n$,
$$\frac{u^n - u^{n+1}}{\Delta t} - \tfrac{\sigma^2}{2}\Delta u^n + \tfrac12|\nabla u^n|^2 = F(x,m^n).$$
The $|\nabla u^n|^2$ term is **nonlinear** in $u^n$. Two options:
1. **Newton inside each timestep** — solve a linearized system at each iterate of the outer fixed-point loop.
2. **Lagged Picard** — at outer iterate $k$ replace $|\nabla u^n|^2 \to \nabla u^{n,(k-1)}\!\cdot\!\nabla u^{n,(k)}$. The product is linear in the unknown $u^{n,(k)}$, so each HJB step becomes an *affine* linear problem — much simpler. At convergence the two factors coincide and we recover $|\nabla u|^2$.

We adopt the lagged-Picard variant. Setting $b := \nabla u^{n,(k-1)}$ (the "old" gradient), the affine HJB step solves:
$$\int_\Omega u^n v + \tfrac{\sigma^2 \Delta t}{2}\nabla u^n\!\cdot\!\nabla v + \tfrac{\Delta t}{2}\,(b\!\cdot\!\nabla u^n)\,v\,dx = \int_\Omega u^{n+1}\,v + \Delta t\,F(m^n)\,v\,dx.$$

### 5.2 FP step (forward Euler implicit)

Given $m^n$ and the just-updated $u^{n+1}$,
$$\int_\Omega m^{n+1} v + \tfrac{\sigma^2 \Delta t}{2}\nabla m^{n+1}\!\cdot\!\nabla v + \Delta t\,m^{n+1}\,\nabla u^{n+1}\!\cdot\!\nabla v\,dx = \int_\Omega m^n\,v\,dx.$$

### 5.3 The Picard loop

```
m^(0)(t,x) = m_0(x)  (constant in time)
for k = 1, …, K_max:
    solve HJB backward  →  u^(k)
    solve FP forward    →  m̃
    m^(k)   = θ m̃ + (1−θ) m^(k−1)
    u_old^(k) = θ u^(k) + (1−θ) u_old^(k−1)
    if ‖m^(k) − m^(k−1)‖_{L²(0,T;L²)} < tol: break
```
Damping (`θ ≈ 0.3`) is essential for the log-congestion case: $F$ is not Lasry–Lions monotone, so the undamped iteration can oscillate. The lagged-Picard linearization of HJB is consistent — at the fixed point the lag vanishes and we have a true MFG solution.

### 5.4 A small numerical detail

`log(m + ε)` can blow up if $m$ briefly dips slightly negative at quadrature points (FE basis functions are not non-negative on every test point even if nodally non-negative). We use `safe_log(x) = log(max(x, 1e-12))` everywhere.
""")

# =============================================================================
# 6. Implementation
# =============================================================================
addmd!(raw"""
## 6. Implementation: HJB step, FP step, Picard loop

Top-level constants (Gaussian initial bump centered at $(0.2,0.5)$, normalized to mass 1; quadratic Hamiltonian; logarithmic congestion):
""")

addcode!(raw"""
const σ_default  = 0.25         # default diffusion coefficient
const σ_art      = 0.05         # artificial viscosity used when stochastic=false
const κ          = 0.10         # congestion strength
const ε_log      = 1e-3         # log regularization
const T_final    = 1.0
const Nt_steps   = 40
const Picard_K   = 15
const Picard_θ   = 0.3
const Picard_tol = 1e-3
const m0_center  = (0.20, 0.50)
const m0_width   = 0.10

@inline safe_log(x) = log(max(x, 1e-12))

\"\"\"
    init_density(S) -> FEFunction in U_m

Project a Gaussian bump centered at `m0_center` (width `m0_width`) onto the
P1 trial space, then renormalize so that ∫_Ω m₀ dx = 1.
\"\"\"
function init_density(S::MFGSpaces)
    bump(x) = exp(-((x[1]-m0_center[1])^2 + (x[2]-m0_center[2])^2)/(2*m0_width^2))
    raw = interpolate_everywhere(bump, S.U_m)
    M0 = sum(∫(raw)*S.dΩ)
    interpolate_everywhere(x -> bump(x)/M0, S.U_m)
end
""")

addmd!(raw"""
**HJB step.** Solves the lagged-linearized backward step for $u^n$:
$$\int_\Omega u\,v + \tfrac{\sigma^2 \Delta t}{2}\nabla u\!\cdot\!\nabla v + \tfrac{\Delta t}{2}(\nabla u_{\text{old}}\!\cdot\!\nabla u)\,v\,dx = \int_\Omega u^{n+1} v + \Delta t\,\kappa\,\log(m^n+\varepsilon)\,v\,dx.$$
""")

addcode!(raw"""
\"\"\"
    solve_hjb_step(S, u_next, m_n, u_prev_n, σ, dt) -> u^n

Lagged-Picard backward implicit-Euler step of the HJB.
\"\"\"
function solve_hjb_step(S::MFGSpaces, u_next, m_n, u_prev_n, σ::Float64, dt::Float64)
    F_m = κ * (safe_log∘(m_n + ε_log))
    a(u, v) = ∫( u*v +
                 (σ^2*dt/2) * (∇(u) ⋅ ∇(v)) +
                 (dt/2)     * ((∇(u_prev_n) ⋅ ∇(u)) * v) ) * S.dΩ
    l(v)    = ∫( u_next*v + dt*F_m*v ) * S.dΩ
    op = AffineFEOperator(a, l, S.U_u, S.V_u)
    return solve(op)
end
""")

addmd!(raw"""
**FP step.** Linear in $m^{n+1}$ once $u^{n+1}$ is fixed:
$$\int_\Omega m\,v + \tfrac{\sigma^2 \Delta t}{2}\nabla m\!\cdot\!\nabla v + \Delta t\,m\,\nabla u^{n+1}\!\cdot\!\nabla v\,dx = \int_\Omega m^n\,v\,dx.$$
""")

addcode!(raw"""
\"\"\"
    solve_fp_step(S, m_prev, u_np1, σ, dt) -> m^{n+1}

Implicit-Euler forward step of the Fokker–Planck equation.
\"\"\"
function solve_fp_step(S::MFGSpaces, m_prev, u_np1, σ::Float64, dt::Float64)
    a(m, v) = ∫( m*v +
                 (σ^2*dt/2) * (∇(m) ⋅ ∇(v)) +
                 dt * (m * (∇(u_np1) ⋅ ∇(v))) ) * S.dΩ
    l(v)    = ∫( m_prev * v ) * S.dΩ
    op = AffineFEOperator(a, l, S.U_m, S.V_m)
    return solve(op)
end
""")

addmd!(raw"""
**Picard loop.** Wraps the backward HJB sweep and forward FP sweep into a damped fixed-point iteration. Returns the full space–time arrays of nodal values for $u$ and $m$ along with diagnostic histories (residual `‖Δm‖_{L²(0,T;L²)}`, total cost).
""")

addcode!(raw"""
struct MFGSolution
    u_traj :: Vector
    m_traj :: Vector
    times  :: Vector{Float64}
    res    :: Vector{Float64}
    cost   :: Vector{Float64}
    σ_eff  :: Float64
    spaces :: MFGSpaces
end

function _l2_st_norm(S::MFGSpaces, A, B, dt)
    r2 = 0.0
    for n in 1:length(A)
        d = A[n] - B[n]
        r2 += sum(∫(d*d)*S.dΩ) * dt
    end
    return sqrt(r2)
end

\"\"\"
    picard_loop(S; σ, T, Nt, K_max, tol, θ, stochastic, verbose) -> MFGSolution

Lagged-Picard fixed-point iteration. With `stochastic=false` (deterministic
agents), σ is replaced by a small artificial viscosity σ_art for stability.
\"\"\"
function picard_loop(S::MFGSpaces; σ::Float64=σ_default, T::Float64=T_final,
                     Nt::Int=Nt_steps, K_max::Int=Picard_K, tol::Float64=Picard_tol,
                     θ::Float64=Picard_θ, stochastic::Bool=true, verbose::Bool=true)
    σ_eff = stochastic ? σ : σ_art
    dt = T/Nt; times = collect(0:Nt) .* dt
    if !stochastic
        verbose && @printf "stochastic=false:  σ ← σ_art = %.3g\n" σ_eff
    end

    m0 = init_density(S)
    m_traj = [FEFunction(S.U_m, copy(get_free_dof_values(m0))) for _ in 1:Nt+1]
    u_prev = [FEFunction(S.U_u, zeros(num_free_dofs(S.U_u))) for _ in 1:Nt+1]
    uT     = FEFunction(S.U_u, zeros(num_free_dofs(S.U_u)))   # g(x,m)=0

    res_hist  = Float64[]; cost_hist = Float64[]
    u_traj = u_prev
    for k in 1:K_max
        # HJB backward sweep
        u_traj = Vector{Any}(undef, Nt+1); u_traj[Nt+1] = uT
        for n in Nt:-1:1
            u_traj[n] = solve_hjb_step(S, u_traj[n+1], m_traj[n], u_prev[n], σ_eff, dt)
        end
        # FP forward sweep
        m_tilde = Vector{Any}(undef, Nt+1); m_tilde[1] = m_traj[1]
        for n in 1:Nt
            m_tilde[n+1] = solve_fp_step(S, m_tilde[n], u_traj[n+1], σ_eff, dt)
        end
        push!(res_hist, _l2_st_norm(S, m_tilde, m_traj, dt))

        # Damped updates
        m_new = [FEFunction(S.U_m, θ*get_free_dof_values(m_tilde[n]) +
                                   (1-θ)*get_free_dof_values(m_traj[n])) for n in 1:Nt+1]
        u_new = [FEFunction(S.U_u, θ*get_free_dof_values(u_traj[n]) +
                                   (1-θ)*get_free_dof_values(u_prev[n])) for n in 1:Nt+1]
        m_traj = m_new; u_prev = u_new

        # cost = ∫₀ᵀ ∫_Ω (½|∇u|² + κ log(m+ε)) m dx dt   (trapezoidal)
        c = 0.0
        for n in 1:Nt+1
            wt = (n==1 || n==Nt+1) ? 0.5*dt : dt
            integrand = (0.5*(∇(u_traj[n])⋅∇(u_traj[n])) +
                         κ*(safe_log∘(m_traj[n]+ε_log))) * m_traj[n]
            c += sum(∫(integrand)*S.dΩ) * wt
        end
        push!(cost_hist, c)
        verbose && @printf "  Picard k=%2d   ‖Δm‖ = %.3e   cost = %.3e\n" k res_hist[end] cost_hist[end]
        res_hist[end] < tol && break
    end
    return MFGSolution(u_traj, m_traj, times, res_hist, cost_hist, σ_eff, S)
end
""")

# =============================================================================
# 7. Default scenario solve
# =============================================================================
addmd!(raw"""
## 7. Default scenario solve

We solve the MFG with $\sigma=0.25$, $\kappa=0.10$, $T=1$, $\Delta t = 0.025$, and Picard damping $\theta=0.3$. With those settings the iteration converges to $\|\Delta m\| < 10^{-2}$ within $\sim15$ iterations on the default mesh. Wall time on a laptop CPU: a few minutes (most of it Gridap precompilation on first run).

A note on $\sigma$: smaller $\sigma$ pushes the FP equation toward advection-dominated regime where central-Galerkin without stabilization (no SUPG) becomes oscillatory. With the default parameters the cell Péclet number $\mathrm{Pe} = h\,|\nabla u|/\sigma^2$ stays $\lesssim 1$ in most of the domain.
""")

addcode!(raw"""
@time sol = picard_loop(S; verbose=true)
@printf "\nFinal residual: %.3e   (tol=%.3e, K=%d iters)\n" sol.res[end] Picard_tol length(sol.res)
@printf "Mass at t=0:    %.5f\n" sum(∫(sol.m_traj[1])*S.dΩ)
@printf "Mass at t=T:    %.5f\n" sum(∫(sol.m_traj[end])*S.dΩ)
""")

# =============================================================================
# 8. Validation cells
# =============================================================================
addmd!(raw"""
## 8. Validation

### 8.1 No-congestion limit ($\kappa = 0$)

When $\kappa = 0$ and there is *no* running cost, the HJB has the trivial solution $u \equiv 0$ (since $u(T)=0$ and $u|_{\Gamma_e}=0$) — uninformative. The standard sanity check uses $\kappa=0$ together with a **unit running cost** $L = \tfrac12|\alpha|^2 + 1$, recovering the classical viscous eikonal
$$-\partial_t u - \tfrac{\sigma^2}{2}\Delta u + \tfrac12|\nabla u|^2 = 1,\quad u(T)=0,\quad u|_{\Gamma_e}=0.$$
Heuristically, $u(t,x)$ is the (viscosity-smoothed) **expected travel time** from $(t,x)$ to the exit; the optimal feedback $-\nabla u$ should point everywhere toward the exit. The Cole–Hopf transform $\phi=\exp(-u/\sigma^2)$ linearizes the spatial part to a heat-like equation, so $u$ is morally a smoothed distance.

We verify qualitatively: $u$ is large far from the exit and rises toward the upper-left corner; $-\nabla u$ points monotonically toward the exit.
""")

addcode!(raw"""
# In the no-congestion limit, F(x,m) = 0 makes u ≡ 0 the only solution
# compatible with u(T)=0 and u|_Γe=0 — not informative.  We instead drop
# the congestion *but keep a unit running cost* L = ½|∇u|² + 1, recovering
# the classical viscous eikonal whose solution is a smoothed distance to Γ_e.
function picard_loop_kappa0(S::MFGSpaces; σ=σ_default, T=T_final, Nt=Nt_steps,
                            K_max=10, θ=Picard_θ)
    dt = T/Nt
    u_prev = [FEFunction(S.U_u, zeros(num_free_dofs(S.U_u))) for _ in 1:Nt+1]
    uT     = FEFunction(S.U_u, zeros(num_free_dofs(S.U_u)))
    function hjb_unitF(u_next, u_prev_n, σ, dt)
        a(u,v) = ∫( u*v + (σ^2*dt/2)*(∇(u)⋅∇(v)) + (dt/2)*((∇(u_prev_n)⋅∇(u))*v) ) * S.dΩ
        l(v)   = ∫( u_next*v + dt*v ) * S.dΩ
        solve(AffineFEOperator(a, l, S.U_u, S.V_u))
    end
    u_traj = u_prev
    for k in 1:K_max
        u_traj = Vector{Any}(undef, Nt+1); u_traj[Nt+1] = uT
        for n in Nt:-1:1
            u_traj[n] = hjb_unitF(u_traj[n+1], u_prev[n], σ, dt)
        end
        u_new = [FEFunction(S.U_u, θ*get_free_dof_values(u_traj[n]) +
                                   (1-θ)*get_free_dof_values(u_prev[n])) for n in 1:Nt+1]
        u_prev = u_new
    end
    return u_traj
end

u_kappa0 = picard_loop_kappa0(S)

# Visualize u_kappa0 at t=0 alongside u(t=0) of full MFG
function sample_field_grid(uh, nx::Int=80, ny::Int=80; bbox=(0.0,1.0,0.0,1.0))
    xs = range(bbox[1], bbox[2]; length=nx)
    ys = range(bbox[3], bbox[4]; length=ny)
    U  = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        try; U[i,j] = uh(Point(xs[i], ys[j])); catch; end
    end
    return collect(xs), collect(ys), U
end

function _safe_range(M)
    v = filter(isfinite, vec(M))
    isempty(v) && return (0.0, 1.0)
    lo, hi = minimum(v), maximum(v)
    hi - lo < 1e-12 && (hi = lo + 1.0)
    return (lo, hi)
end

function _clean!(M)
    @inbounds for k in eachindex(M); isfinite(M[k]) || (M[k] = 0.0); end
    M
end

let
    xs, ys, U0 = sample_field_grid(u_kappa0[1]); _clean!(U0)
    _,  _,  Uf = sample_field_grid(sol.u_traj[1]); _clean!(Uf)
    r0 = _safe_range(U0); rf = _safe_range(Uf)
    fig = Figure(; size=(820, 380))
    ax1 = Axis(fig[1,1]; aspect=DataAspect(), title="u(t=0)  [κ=0, viscous eikonal w/ unit cost]")
    hm1 = heatmap!(ax1, xs, ys, U0; colormap=:viridis, colorrange=r0)
    Colorbar(fig[1,2], hm1)
    ax2 = Axis(fig[1,3]; aspect=DataAspect(), title="u(t=0)  [κ=$(κ), full MFG]")
    hm2 = heatmap!(ax2, xs, ys, Uf; colormap=:viridis, colorrange=rf)
    Colorbar(fig[1,4], hm2)
    save(joinpath(OUTPUTS, "validation_kappa0.png"), fig)
    fig
end
""")

addmd!(raw"""
### 8.2 Mass conservation / exit flux

For absorbing exit BC $m=0$ on $\Gamma_e$, the outflow is purely diffusive:
$$\frac{dM}{dt} = -\oint_{\partial\Omega} J\!\cdot\!\nu\,dS = \int_{\Gamma_e}\!\tfrac{\sigma^2}{2}\,\partial_\nu m\,dS \;\le\; 0,$$
since $m=0$ at the exit and $\partial_\nu m < 0$ (gradient points inward). The wall contribution vanishes by the no-flux BC. We compare the discrete mass loss to the FE evaluation of the exit flux.

> *Note.* The validation as initially stated involves $\int_{\Gamma_e} m(-\nabla u)\!\cdot\!\nu$, but with strict Dirichlet $m=0$ this advective term vanishes identically. The physically meaningful flux is the diffusive one above. With $\sigma=0$ (Robin/free-outflow BC) the advective term is the right one — an alternative discretization we do not pursue here.
""")

addcode!(raw"""
times = sol.times
masses = [sum(∫(m)*S.dΩ) for m in sol.m_traj]
σ² = sol.σ_eff^2
flux_exit = [sum(∫((σ²/2)*(∇(m)⋅sol.spaces.n_Γe))*S.dΓe) for m in sol.m_traj]
# dM/dt by central differences
dM_dt = similar(masses)
for n in 2:length(masses)-1
    dM_dt[n] = (masses[n+1] - masses[n-1])/(times[n+1] - times[n-1])
end
dM_dt[1]   = (masses[2] - masses[1])/(times[2] - times[1])
dM_dt[end] = (masses[end] - masses[end-1])/(times[end] - times[end-1])

let
    fig = Figure(; size=(720, 320))
    ax1 = Axis(fig[1,1]; xlabel="t", ylabel="∫_Ω m dx", title="Total mass vs t")
    lines!(ax1, times, masses; linewidth=2)
    ax2 = Axis(fig[1,2]; xlabel="t", ylabel="rate", title="dM/dt vs computed exit flux")
    lines!(ax2, times, dM_dt;     linewidth=2, label="dM/dt (FD)")
    lines!(ax2, times, flux_exit; linewidth=2, linestyle=:dash, label="∫_Γ_e (σ²/2) ∂_ν m dS")
    axislegend(ax2; position=:rb)
    save(joinpath(OUTPUTS, "validation_mass.png"), fig)
    fig
end
""")

addmd!(raw"""
### 8.3 Cost decrease across Picard iterations

We expect the running cost
$$\mathcal{J} = \int_0^T\!\!\int_\Omega \big(\tfrac12|\nabla u|^2 + \kappa\log(m+\varepsilon)\big)\,m\,dx\,dt$$
to decrease (and eventually plateau) along the iteration, modulo small fluctuations from the damping and from $F$ being non-monotone in $m$.

### 8.4 Picard residual

A log-scale plot of $\|m^{(k)} - m^{(k-1)}\|_{L^2(0,T;L^2)}$ vs. $k$.
""")

addcode!(raw"""
let
    fig = Figure(; size=(720, 320))
    ax1 = Axis(fig[1,1]; xlabel="Picard iter k", ylabel="cost", title="Total cost across Picard iterations")
    scatterlines!(ax1, 1:length(sol.cost), sol.cost; linewidth=2)
    ax2 = Axis(fig[1,2]; xlabel="Picard iter k", ylabel="‖m^(k) - m^(k-1)‖", yscale=log10,
               title="Picard residual (log-scale)")
    scatterlines!(ax2, 1:length(sol.res), sol.res; linewidth=2)
    save(joinpath(OUTPUTS, "validation_picard.png"), fig)
    fig
end
""")

# =============================================================================
# 9. Visualization
# =============================================================================
addmd!(raw"""
## 9. Visualization

### 9.1 Static snapshots: $m(t,x)$ and $u(t,x)$ at five times

A 2×5 grid showing $m$ on top and $u$ on bottom, sampled to a regular grid for `heatmap`. Same color scale within each row.
""")

addcode!(raw"""
let
    nx, ny = 100, 100
    snap_times = [0.0, T_final/4, T_final/2, 3T_final/4, T_final]
    snap_idx = [argmin(abs.(sol.times .- t)) for t in snap_times]

    function _grid(uh)
        G = sample_field_grid(uh, nx, ny)[3]
        @inbounds for k in eachindex(G); isfinite(G[k]) || (G[k] = 0.0); end
        G
    end
    M_snaps = [_grid(sol.m_traj[i]) for i in snap_idx]
    U_snaps = [_grid(sol.u_traj[i]) for i in snap_idx]
    mlim = (0.0, max(1e-9, maximum(maximum(M) for M in M_snaps)))
    ulim = (0.0, max(1e-9, maximum(maximum(U) for U in U_snaps)))
    xs = range(0,1; length=nx); ys = range(0,1; length=ny)

    fig = Figure(; size=(1200, 540))
    for (col, (i, t)) in enumerate(zip(snap_idx, snap_times))
        ax_m = Axis(fig[1, col]; aspect=DataAspect(), title="m(t=$(round(t; digits=2)))")
        heatmap!(ax_m, xs, ys, M_snaps[col]; colormap=:viridis, colorrange=mlim)
        lines!(ax_m, [1,1], [exit_y_def...]; color=:red, linewidth=2)
        ax_u = Axis(fig[2, col]; aspect=DataAspect(), title="u(t=$(round(t; digits=2)))")
        heatmap!(ax_u, xs, ys, U_snaps[col]; colormap=:plasma, colorrange=ulim)
        lines!(ax_u, [1,1], [exit_y_def...]; color=:cyan, linewidth=2)
    end
    Colorbar(fig[1, 6], colormap=:viridis, limits=mlim)
    Colorbar(fig[2, 6], colormap=:plasma,  limits=ulim)
    save(joinpath(OUTPUTS, "snapshots.png"), fig)
    fig
end
""")

addmd!(raw"""
### 9.2 Animated density with $-\nabla u$ quiver overlay

We `record` an `mp4` of $m(t,x)$ evolving, with a coarse arrow grid showing the optimal control field $-\nabla u(t,x)$ at each frame.
""")

addcode!(raw"""
# For each grid point, returns a flat vector of [base, tip, base, tip, ...]
# suitable for `linesegments!`. Avoids Makie's deprecated arrows recipe entirely.
function _quiver_segments(uh, qxs, qys; clip::Float64=0.04)
    segs = Point2f[]
    for j in 1:length(qys), i in 1:length(qxs)
        x = qxs[i]; y = qys[j]
        local g
        try
            g = (∇(uh))(Point(x, y))
        catch
            continue
        end
        gn = sqrt(g[1]^2 + g[2]^2)
        gn < 1e-8 && continue
        sc = clip/(1 + 0.3*gn)
        dx = -g[1]*sc; dy = -g[2]*sc
        push!(segs, Point2f(x, y))
        push!(segs, Point2f(x + dx, y + dy))
    end
    return segs
end

function animate_density(sol::MFGSolution; path::String, nx::Int=80, ny::Int=80,
                         qx::Int=14, qy::Int=14, fps::Int=15)
    xs, ys, _ = sample_field_grid(sol.m_traj[1], nx, ny)
    qxs = collect(range(0.05, 0.95; length=qx))
    qys = collect(range(0.05, 0.95; length=qy))
    nframes = length(sol.times)

    # Pre-compute every frame's density grid (cheap relative to encoding) and a
    # robust color range; replace NaNs with 0 so CairoMakie's image rendering
    # doesn't see any non-finite values.
    M_frames = Vector{Matrix{Float64}}(undef, nframes)
    cmax = 0.0
    for i in 1:nframes
        Mi = sample_field_grid(sol.m_traj[i], nx, ny)[3]
        @inbounds for k in eachindex(Mi)
            isfinite(Mi[k]) || (Mi[k] = 0.0)
        end
        M_frames[i] = Mi
        cmax = max(cmax, maximum(Mi))
    end
    cmax = max(cmax, 1e-9)
    cr = (0.0, cmax)

    M_obs    = Observable(M_frames[1])
    seg_obs  = Observable(_quiver_segments(sol.u_traj[1], qxs, qys))
    tip_obs  = Observable([s for (k,s) in enumerate(_quiver_segments(sol.u_traj[1], qxs, qys)) if iseven(k)])
    title_obs = Observable("t = 0.000")
    fig = Figure(; size=(560, 540))
    ax  = Axis(fig[1,1]; aspect=DataAspect(), title=title_obs,
               limits=((0,1), (0,1)))
    hm  = heatmap!(ax, xs, ys, M_obs; colormap=:viridis, colorrange=cr)
    Colorbar(fig[1,2], hm)
    linesegments!(ax, seg_obs; color=:white, linewidth=1.2)
    scatter!(ax, tip_obs; color=:white, markersize=4)
    lines!(ax, [1,1], [exit_y_def...]; color=:red, linewidth=2)

    record(fig, path, 1:nframes; framerate=fps) do idx
        M_obs[] = M_frames[idx]
        s = _quiver_segments(sol.u_traj[idx], qxs, qys)
        seg_obs[] = s
        tip_obs[] = [p for (k,p) in enumerate(s) if iseven(k)]
        title_obs[] = @sprintf "t = %.3f" sol.times[idx]
    end
    return path
end

dens_path = joinpath(OUTPUTS, "density_quiver.mp4")
animate_density(sol; path=dens_path)
println("wrote ", dens_path)
""")

addmd!(raw"""
### 9.3 Lagrangian agents over the density

We rejection-sample $\sim 200$ initial agent positions from $m_0$, then push each through the closed-loop SDE
$$dX_t = -\nabla u(t,X_t)\,dt + \sigma\,dB_t$$
with Euler–Maruyama, removing agents that cross the exit. The animation overlays Lagrangian particles on the Eulerian density heatmap. Two runs: stochastic ($\sigma>0$) and deterministic ($\sigma=0$).
""")

addcode!(raw"""
\"\"\"
    sample_initial(S, m0, n; bbox, rng) -> Vector{Tuple{Float64,Float64}}

Rejection-sample n initial positions from the (unnormalized) density m0.
\"\"\"
function sample_initial(S::MFGSpaces, m0, n::Int;
                        bbox=(0.0,1.0,0.0,1.0), rng=Random.default_rng())
    pts = Vector{Tuple{Float64,Float64}}(undef, n)
    xmin, xmax, ymin, ymax = bbox
    Mmax = 1.05 * maximum(get_free_dof_values(m0))
    i = 1
    while i ≤ n
        x = xmin + (xmax-xmin)*rand(rng); y = ymin + (ymax-ymin)*rand(rng)
        local val
        try; val = m0(Point(x,y)); catch; continue; end
        if val > 0 && rand(rng)*Mmax < val
            pts[i] = (x, y); i += 1
        end
    end
    return pts
end

\"\"\"
    simulate_agents(sol; n_agents, stochastic, rng) -> Vector{Matrix{Float64}}

Each trajectory is an (Nt+1)×2 matrix; after exit, rows are NaN.
\"\"\"
function simulate_agents(sol::MFGSolution; n_agents::Int=200,
                         stochastic::Bool=true, rng=Random.default_rng())
    S = sol.spaces; times = sol.times; Nt = length(times)-1
    dt = times[2] - times[1]
    σ_use = stochastic ? sol.σ_eff : 0.0
    init = sample_initial(S, sol.m_traj[1], n_agents; rng=rng)
    traj = [fill(NaN, Nt+1, 2) for _ in 1:n_agents]
    alive = trues(n_agents)
    for i in 1:n_agents
        traj[i][1, :] .= init[i]
    end
    for n in 1:Nt
        u_h = sol.u_traj[n]
        for i in 1:n_agents
            alive[i] || continue
            x, y = traj[i][n,1], traj[i][n,2]
            local g
            try; g = (∇(u_h))(Point(x,y)); catch; alive[i]=false; continue; end
            dx = -g[1]*dt; dy = -g[2]*dt
            if stochastic
                dx += σ_use*sqrt(dt)*randn(rng)
                dy += σ_use*sqrt(dt)*randn(rng)
            end
            xn = clamp(x+dx, 1e-6, 1 - 1e-6)
            yn = clamp(y+dy, 1e-6, 1 - 1e-6)
            if xn ≥ 1 - 2e-6 && exit_y_def[1] ≤ yn ≤ exit_y_def[2]
                alive[i] = false
            else
                traj[i][n+1,1] = xn; traj[i][n+1,2] = yn
            end
        end
    end
    return traj
end

function animate_agents(sol::MFGSolution, traj; path::String,
                        nx::Int=80, ny::Int=80, fps::Int=15)
    xs, ys, _ = sample_field_grid(sol.m_traj[1], nx, ny)
    nframes = length(sol.times); n_agents = length(traj)

    # Pre-compute frames and robust colorrange.
    M_frames = Vector{Matrix{Float64}}(undef, nframes)
    cmax = 0.0
    for i in 1:nframes
        Mi = sample_field_grid(sol.m_traj[i], nx, ny)[3]
        @inbounds for k in eachindex(Mi)
            isfinite(Mi[k]) || (Mi[k] = 0.0)
        end
        M_frames[i] = Mi
        cmax = max(cmax, maximum(Mi))
    end
    cr = (0.0, max(cmax, 1e-9))

    M_obs = Observable(M_frames[1])
    pts_obs = Observable(Point2f[])
    title_obs = Observable("t = 0.000")
    fig = Figure(; size=(560, 540))
    ax  = Axis(fig[1,1]; aspect=DataAspect(), title=title_obs,
               limits=((0,1), (0,1)))
    hm  = heatmap!(ax, xs, ys, M_obs; colormap=:viridis, colorrange=cr)
    Colorbar(fig[1,2], hm)
    scatter!(ax, pts_obs; color=:orange, markersize=5, strokecolor=:black, strokewidth=0.4)
    lines!(ax, [1,1], [exit_y_def...]; color=:red, linewidth=2)

    record(fig, path, 1:nframes; framerate=fps) do idx
        M_obs[] = M_frames[idx]
        pts = Point2f[]
        for i in 1:n_agents
            x, y = traj[i][idx,1], traj[i][idx,2]
            isnan(x) && continue
            push!(pts, Point2f(x, y))
        end
        pts_obs[] = pts
        title_obs[] = @sprintf "t = %.3f" sol.times[idx]
    end
    return path
end

# Stochastic run.  We use 100 agents in the notebook so the cell runs in
# reasonable wall time when executed via nbconvert; bump to 200+ for richer
# overlays when running interactively.
Random.seed!(42)
@time traj_st = simulate_agents(sol; n_agents=100, stochastic=true)
println("simulated $(length(traj_st)) trajectories"); flush(stdout)
animate_agents(sol, traj_st; path=joinpath(OUTPUTS, "agents_stochastic.mp4"), nx=60, ny=60)
println("wrote agents_stochastic.mp4"); flush(stdout)
""")

# =============================================================================
# 10. σ=0 deterministic case
# =============================================================================
addmd!(raw"""
## 10. Deterministic case ($\sigma=0$)

When $\sigma=0$, the FP equation becomes a pure transport equation, and the HJB becomes a first-order eikonal-type problem. Standard central-Galerkin without stabilization is *not* coercive in this regime, so following the problem statement we add a small **artificial viscosity** $\sigma_{\text{art}} = 0.05$. Larger than the spec's $10^{-3}$, but with $h\sim 0.04$ and $|\nabla u|\sim O(1)$ the cell Péclet number $\mathrm{Pe} \sim h/\sigma_{\text{art}}^2$ is unforgiving for very small $\sigma_{\text{art}}$; without SUPG/upwinding, the iteration diverges. Production-quality first-order MFG codes use monotone finite-difference schemes (Achdou–Capuzzo-Dolcetta) or SUPG/DG.

The agents are pushed by Euler with $\sigma=0$ (purely deterministic flow along $-\nabla u$).
""")

addcode!(raw"""
@time sol_det = picard_loop(S; stochastic=false, verbose=true)
@printf "deterministic mass loss: %.4f → %.4f over [0,T]\n" sum(∫(sol_det.m_traj[1])*S.dΩ) sum(∫(sol_det.m_traj[end])*S.dΩ)
flush(stdout)

Random.seed!(42)
traj_det = simulate_agents(sol_det; n_agents=100, stochastic=false)
println("simulated $(length(traj_det)) deterministic trajectories"); flush(stdout)
animate_agents(sol_det, traj_det; path=joinpath(OUTPUTS, "agents_deterministic.mp4"), nx=60, ny=60)
println("wrote agents_deterministic.mp4"); flush(stdout)
""")

# =============================================================================
# 11. Loading custom mesh
# =============================================================================
addmd!(raw"""
## 11. Loading a custom `.msh`

To run on your own geometry, save a `.msh` file with **physical groups** named `wall` (no-flux) and `exit` (Dirichlet) on the relevant boundary curves, and pass the path to `load_mesh`. Below is a placeholder cell — uncomment and edit:

```julia
# my_mesh = load_mesh("path/to/your_room.msh")
# my_S    = build_spaces(my_mesh)
# my_sol  = picard_loop(my_S; verbose=true)
```

You may also need to adjust `m0_center`, `m0_width`, the bounding box passed to `sample_initial`, and the agent-domain clamps in `simulate_agents` so they fit your geometry (the defaults assume the unit square).
""")

addcode!(raw"""
# placeholder — edit the path below to use your own .msh file
# my_path = "path/to/your_room.msh"
# my_model = load_mesh(my_path)
# my_S = build_spaces(my_model)
# my_sol = picard_loop(my_S)
println("(custom-mesh placeholder cell — see comments above)")
""")

# =============================================================================
# 12. Discussion
# =============================================================================
addmd!(raw"""
## 12. Discussion

### Limitations of the lagged-Picard scheme

* **No global convergence guarantees for log-congestion.** Logarithmic running cost $F(x,m) = \kappa\log(m+\varepsilon)$ is a *decreasing* function of $m$ near $m\!\to\!0^+$, so it violates the Lasry–Lions monotonicity that gives uniqueness of the MFG equilibrium and contractivity of the Picard map. Damping is essential; even with damping, the iteration may stall or settle in a non-unique fixed point.
* **Lag introduces a one-iterate error that vanishes only at the fixed point.** While each HJB sub-problem is linear, the lag term $\nabla u_{\text{old}}\!\cdot\!\nabla u_{\text{new}}$ equals the true $|\nabla u|^2$ only when the iteration has converged. Convergence is linear in the residual.
* **Galerkin advection.** Without SUPG, the FP equation is unstable for cell Péclet $\gtrsim 1$. We compensate by choosing $\sigma$ moderately large and $h$ small. SUPG/DG would let us push to small $\sigma$ cleanly.

### What changes for first-order MFG ($\sigma=0$)?

* The HJB becomes a **viscosity solution** of $-\partial_t u + \tfrac12|\nabla u|^2 = F$; standard FE without monotonicity-preserving stabilization can pick non-physical shocks/branches.
* The FP becomes a transport equation $\partial_t m + \nabla\!\cdot\!(m\,\alpha^*) = 0$. Density can develop **caustics/shocks**.
* Achdou–Capuzzo-Dolcetta semi-Lagrangian or upwind FD schemes are the standard remedy.
* Our deterministic case sidesteps both issues by adding a tiny $\sigma_{\text{art}}$ and reusing the parabolic machinery — pragmatic, not rigorous.

### Extensions

* Newton inside each HJB step, replacing the lag, gives quadratic local convergence.
* Anisotropic congestion or non-quadratic Hamiltonians fit the same skeleton.
* Time-dependent obstacles → re-mesh per time-step or add a penalty term.
* Multi-population MFG → couple multiple $(u,m)$ pairs through a joint $F$.

### Files written under `outputs/`

* `mesh.png` — default mesh
* `validation_kappa0.png`, `validation_mass.png`, `validation_picard.png` — validation cells
* `snapshots.png` — static $m,u$ at five times
* `density_quiver.mp4` — animated density + control quiver
* `agents_stochastic.mp4`, `agents_deterministic.mp4` — Lagrangian particle overlays
* `room.msh` — the generated default mesh

Mathematical and numerical references: Lasry–Lions 2007 (foundations); Achdou–Capuzzo-Dolcetta 2010 (FD schemes); Carlini–Silva 2014 (semi-Lagrangian); Briceño-Arias–Kalise–Silva 2018 (proximal/optim); Cardaliaguet's lecture notes 2013.
""")

# =============================================================================
# Build .ipynb
# =============================================================================
function build_cell(kind::Symbol, src::String)
    src_lines = split(src, '\n', keepempty=true)
    # Each line gets a trailing \n except the last
    lines = String[]
    for (i, ln) in enumerate(src_lines)
        push!(lines, i == length(src_lines) ? ln : ln*"\n")
    end
    cell = Dict{String,Any}(
        "cell_type" => kind == :md ? "markdown" : "code",
        "id" => string(uuid4())[1:8],
        "metadata" => Dict{String,Any}(),
        "source" => lines,
    )
    if kind == :code
        cell["execution_count"] = nothing
        cell["outputs"] = Any[]
    end
    return cell
end

nb = Dict{String,Any}(
    "cells" => [build_cell(k, s) for (k, s) in CELLS],
    "metadata" => Dict{String,Any}(
        "kernelspec" => Dict{String,Any}(
            "display_name" => "Julia 1.12",
            "language"     => "julia",
            "name"         => "julia-1.12",
        ),
        "language_info" => Dict{String,Any}(
            "name"           => "julia",
            "file_extension" => ".jl",
            "mimetype"       => "application/julia",
            "version"        => "1.12.5",
        ),
    ),
    "nbformat" => 4,
    "nbformat_minor" => 5,
)

open(NOTEBOOK_PATH, "w") do io
    JSON.print(io, nb, 1)
end

println("Wrote ", NOTEBOOK_PATH, "  (", length(CELLS), " cells)")
