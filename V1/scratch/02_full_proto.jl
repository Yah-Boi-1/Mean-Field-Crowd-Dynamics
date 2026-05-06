using Gmsh
using Gridap, GridapGmsh
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random

# -----------------------------------------------------------------------------
# Mesh
# -----------------------------------------------------------------------------
function build_room_mesh(path::String; h::Float64=0.05,
                         exit_y::Tuple{Float64,Float64}=(0.45, 0.55))
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

path = joinpath(@__DIR__, "room.msh")
build_room_mesh(path; h=0.04)
model = GmshDiscreteModel(path)
labels = get_face_labeling(model)
println("Tags found: ", labels.tag_to_name)

# -----------------------------------------------------------------------------
# FE spaces. P1 Lagrange. Dirichlet on "exit" for both u and m.
# -----------------------------------------------------------------------------
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V_u = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["exit"])
U_u = TrialFESpace(V_u, 0.0)
V_m = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["exit"])
U_m = TrialFESpace(V_m, 0.0)
println("u dofs (free): ", num_free_dofs(V_u),
        " | m dofs (free): ", num_free_dofs(V_m))

Ω = Triangulation(model)
dΩ = Measure(Ω, 2)

Γe = BoundaryTriangulation(model; tags="exit")
dΓe = Measure(Γe, 2)
n_Γe = get_normal_vector(Γe)

# -----------------------------------------------------------------------------
# Initial / terminal data
# -----------------------------------------------------------------------------
const σ_default  = 0.1
const κ          = 0.1
const ε_log      = 1e-3
const T          = 1.0
const Nt         = 40

# safe_log clamps from below to avoid log(0) or log(negative-small) at quadrature
# points where the FE field can dip below zero by O(machine eps).
@inline safe_log(x) = log(max(x, 1e-12))

m0_unnorm(x) = exp(-((x[1]-0.2)^2 + (x[2]-0.5)^2)/(2*0.1^2))
m0_h_un = interpolate_everywhere(m0_unnorm, U_m)
mass0 = sum(∫(m0_h_un)dΩ)
m0_h = interpolate_everywhere(x -> m0_unnorm(x)/mass0, U_m)
@printf "m0 mass after normalization = %.6f\n" sum(∫(m0_h)dΩ)

# Terminal value g = 0 (zero terminal cost)
uT_h = interpolate_everywhere(x -> 0.0, U_u)

# -----------------------------------------------------------------------------
# HJB step (lagged Picard, backward Euler)
#   Solve for u^n given u^{n+1} (=u_next), m^n (=m_n), and previous Picard u^n_prev
#   u - σ²Δt/2 · Δu + (Δt/2) ∇u_old · ∇u = u_next + Δt · F(m_n)
# -----------------------------------------------------------------------------
function solve_hjb_step(u_next::FEFunction, m_n::FEFunction,
                        u_prev::FEFunction, σ::Float64, dt::Float64)
    F_m = κ * (safe_log∘(m_n + ε_log))
    a(u, v) = ∫( u*v +
                 (σ^2*dt/2) * (∇(u) ⋅ ∇(v)) +
                 (dt/2)     * ((∇(u_prev) ⋅ ∇(u)) * v) ) * dΩ
    l(v)   = ∫( u_next*v + dt * F_m * v ) * dΩ
    op = AffineFEOperator(a, l, U_u, V_u)
    return solve(op)
end

# -----------------------------------------------------------------------------
# Fokker–Planck step (forward Euler implicit)
#   m^{n+1} given m^n, u^{n+1}
#   m + σ²Δt/2 ∇m·∇v + Δt ∫ m ∇u · ∇v  = m^n
# -----------------------------------------------------------------------------
function solve_fp_step(m_prev::FEFunction, u_np1::FEFunction,
                       σ::Float64, dt::Float64)
    a(m, v) = ∫( m*v +
                 (σ^2*dt/2) * (∇(m) ⋅ ∇(v)) +
                 dt * (m * (∇(u_np1) ⋅ ∇(v))) ) * dΩ
    l(v)    = ∫( m_prev * v ) * dΩ
    op = AffineFEOperator(a, l, U_m, V_m)
    return solve(op)
end

# -----------------------------------------------------------------------------
# Backward HJB sweep over time
# -----------------------------------------------------------------------------
function hjb_backward(m_traj::Vector{<:FEFunction}, u_prev_traj::Vector{<:FEFunction};
                      σ::Float64, dt::Float64, Nt::Int, uT::FEFunction)
    u_traj = Vector{FEFunction}(undef, Nt+1)
    u_traj[Nt+1] = uT
    for n in Nt:-1:1
        u_traj[n] = solve_hjb_step(u_traj[n+1], m_traj[n], u_prev_traj[n], σ, dt)
    end
    return u_traj
end

function fp_forward(u_traj::Vector{<:FEFunction}, m0::FEFunction;
                    σ::Float64, dt::Float64, Nt::Int)
    m_traj = Vector{FEFunction}(undef, Nt+1)
    m_traj[1] = m0
    for n in 1:Nt
        m_traj[n+1] = solve_fp_step(m_traj[n], u_traj[n+1], σ, dt)
    end
    return m_traj
end

function l2_norm(f::FEFunction)
    sqrt(sum(∫(f*f)dΩ))
end

function picard_loop(; σ::Float64=σ_default, T::Float64=T, Nt::Int=Nt,
                     K_max::Int=15, tol::Float64=1e-3, θ::Float64=0.5,
                     verbose::Bool=true)
    dt = T/Nt
    σ_eff = σ
    if σ == 0.0
        σ_eff = 1e-3   # tiny artificial viscosity
        verbose && println("σ=0: using artificial viscosity σ_art = ", σ_eff)
    end

    # Init m^(0)(t,·) = m0 for all t; u_prev = 0
    m_traj = [interpolate_everywhere(x -> m0_unnorm(x)/mass0, U_m) for _ in 1:(Nt+1)]
    u_prev = [interpolate_everywhere(x -> 0.0, U_u) for _ in 1:(Nt+1)]

    res_hist = Float64[]
    cost_hist = Float64[]
    u_traj = u_prev   # placeholder

    for k in 1:K_max
        # HJB backward with lagged ∇u from u_prev
        u_traj = hjb_backward(m_traj, u_prev; σ=σ_eff, dt=dt, Nt=Nt, uT=uT_h)
        # diagnostic
        if verbose
            unorms = [l2_norm(u_traj[n]) for n in [1, Nt÷2+1, Nt+1]]
            @printf "  HJB ‖u‖₂ at t=0,T/2,T: %.3e %.3e %.3e\n" unorms[1] unorms[2] unorms[3]
        end
        # FP forward with new u
        m_tilde = fp_forward(u_traj, m_traj[1]; σ=σ_eff, dt=dt, Nt=Nt)
        if verbose
            mnorms = [l2_norm(m_tilde[n]) for n in [1, Nt÷2+1, Nt+1]]
            @printf "  FP  ‖m‖₂ at t=0,T/2,T: %.3e %.3e %.3e\n" mnorms[1] mnorms[2] mnorms[3]
        end

        # Damped update on m_traj (and u_prev for next iterate's lag)
        # residual on m
        r2 = 0.0
        for n in 1:(Nt+1)
            d = m_tilde[n] - m_traj[n]
            r2 += sum(∫(d*d)dΩ)*dt
        end
        push!(res_hist, sqrt(r2))

        m_new = Vector{FEFunction}(undef, Nt+1)
        for n in 1:(Nt+1)
            v = θ*get_free_dof_values(m_tilde[n]) + (1-θ)*get_free_dof_values(m_traj[n])
            m_new[n] = FEFunction(U_m, v)
        end
        m_traj = m_new
        # Damp u_prev too (helps convergence with lagged scheme)
        u_new_traj = Vector{FEFunction}(undef, Nt+1)
        for n in 1:(Nt+1)
            v = θ*get_free_dof_values(u_traj[n]) + (1-θ)*get_free_dof_values(u_prev[n])
            u_new_traj[n] = FEFunction(U_u, v)
        end
        u_prev = u_new_traj

        # cost = ∫₀ᵀ ∫_Ω (½|∇u|² + κ log(m+ε)) m dx dt
        c = 0.0
        for n in 1:(Nt+1)
            wt = (n==1 || n==Nt+1) ? 0.5*dt : dt
            integrand = (0.5*(∇(u_traj[n])⋅∇(u_traj[n])) + κ*(safe_log∘(m_traj[n]+ε_log))) * m_traj[n]
            c += sum(∫(integrand)dΩ) * wt
        end
        push!(cost_hist, c)

        verbose && @printf "Picard iter %2d: ||Δm||₂ = %.4e, cost = %.4e\n" k res_hist[end] cost_hist[end]
        if res_hist[end] < tol
            verbose && println("Converged.")
            break
        end
    end
    return (u=u_traj, m=m_traj, res=res_hist, cost=cost_hist, dt=dt, σ=σ_eff)
end

@time sol = picard_loop(σ=0.25, K_max=15, tol=1e-3, θ=0.3, verbose=true)
println("Done. final res=", sol.res[end])

# Quick sanity: total mass over time
println("\nMass trajectory:")
for n in 1:5:length(sol.m)
    M = sum(∫(sol.m[n])dΩ)
    @printf "t=%.3f mass=%.4f\n" (n-1)*sol.dt M
end

# Exit flux at final time using boundary measure
n_idx = length(sol.m)
σ² = sol.σ^2
flux = sum(∫( (σ²/2) * (∇(sol.m[n_idx]) ⋅ n_Γe) )dΓe)  # diffusive component (m=0 on exit ⇒ advective vanishes)
@printf "Exit flux at t=T: %.4e (should be ≥ 0 for outflow)\n" flux
