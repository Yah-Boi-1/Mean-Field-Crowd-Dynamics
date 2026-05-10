# Mean Field Games and Single-Agent Optimal Control — Julia/Gridap notebooks

This folder contains **two** companion Julia notebooks that share the same
mesh / FEM stack but solve different problems:

1. [`mfg_crowd_dynamics.ipynb`](mfg_crowd_dynamics.ipynb) — a 2D **second-order
   Mean Field Game** (coupled HJB ↔ Fokker–Planck) on an unstructured
   triangle mesh, with population-level density evolution and Picard fixed
   point.
2. [`tracer_agent.ipynb`](tracer_agent.ipynb) — a **single-agent optimal
   control** problem on the same domain. The agent navigates a *prescribed,
   exogenous, time-independent* density field $m(x)$ to reach the exit,
   following the optimal feedback derived from a backward HJB. **Not** an MFG
   — there is one agent, no population dynamics, no fixed point.

The math, derivations, and discussion live inside the notebooks. This README
covers how to run them.

## Quick start

1. **Julia 1.10 or newer.** Tested with **Julia 1.12.5** on Windows 11.
2. Clone this folder. From a shell in the repo root:

   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()           # first run only — installs everything in Manifest.toml
   using IJulia
   notebook(dir=".")           # opens the Julia kernel in a browser tab
   ```
   Then open `mfg_crowd_dynamics.ipynb` or `tracer_agent.ipynb` and Run All.

   Alternatively, headless run from the shell:
   ```bash
   jupyter nbconvert --to notebook --execute mfg_crowd_dynamics.ipynb \
       --output mfg_crowd_dynamics.ipynb --ExecutePreprocessor.timeout=1800
   jupyter nbconvert --to notebook --execute tracer_agent.ipynb \
       --output tracer_agent.ipynb --ExecutePreprocessor.timeout=900
   ```

## What the notebook does

* **§1 setup** — activates the local `Project.toml`/`Manifest.toml` and imports.
* **§2 math** — derives HJB from dynamic programming, the Hamiltonian via
  Legendre transform, FP from the closed-loop SDE, the variational forms
  used by the FEM, and the Picard fixed point.
* **§3 mesh** — `build_default_mesh` scripts a unit-square room with a
  narrow exit on the right wall via [Gmsh](https://gmsh.info), tagged
  `"wall"` (no-flux) and `"exit"` (Dirichlet). `load_mesh(path)` accepts
  any `.msh` you provide with the same physical-group names.
* **§4 weak forms** — derivation of the variational HJB and FP weak forms.
* **§5 lagged Picard** — implicit Euler in time, lagged $|\nabla u|^2$
  to keep each HJB sub-problem linear.
* **§6 implementation** — `solve_hjb_step`, `solve_fp_step`, `picard_loop`
  with damping.
* **§7 default solve** — runs the MFG with σ=0.25, κ=0.10, T=1, Nt=40.
* **§8 validation** — four checks: no-congestion limit, mass conservation,
  cost decrease, Picard residual.
* **§9 visualization** — static snapshot grid, animated density + control
  quiver, Lagrangian agent overlay (stochastic and deterministic).
* **§10 σ=0** — re-solve and re-animate with deterministic agents.
* **§11 custom mesh** — placeholder cell to swap in a user-supplied `.msh`.
* **§12 discussion** — limitations of lagged-Picard, monotonicity, what
  would change for first-order MFG with a monotone scheme.

## Generated artifacts

All saved under `outputs/`:

| File                          | Section | Description                                |
| ----------------------------- | ------- | ------------------------------------------ |
| `room.msh`                    | §3      | Default Gmsh-generated mesh                |
| `mesh.png`                    | §3      | Triangulation with exit highlighted        |
| `validation_kappa0.png`       | §8.1    | $u(0,x)$ for κ=0 vs full MFG               |
| `validation_mass.png`         | §8.2    | Mass-vs-time and exit flux                 |
| `validation_picard.png`       | §8.3–4  | Cost and residual vs Picard iteration      |
| `snapshots.png`               | §9.1    | $m(t,x)$ and $u(t,x)$ at five times        |
| `density_quiver.mp4`          | §9.2    | Animated density with $-\nabla u$ arrows   |
| `agents_stochastic.mp4`       | §9.3    | 200 Lagrangian agents with σ>0             |
| `agents_deterministic.mp4`    | §10     | 200 Lagrangian agents with σ=0             |

## What `tracer_agent.ipynb` does

* **§1 setup** — activates the local project; imports.
* **§2 problem statement** — derives the HJB from dynamic programming,
  documents the artificial-viscosity stability bound, and explains the
  well-posedness condition $F(x_0) > 2L^2/T^2$ that makes the agent prefer
  reaching the exit over lingering on the finite horizon.
* **§3 mesh** — reuses `build_default_mesh` (unit square with right-wall
  exit slit at $\{1\}\times[0.45,0.55]$), $h=0.05$.
* **§4 density field $m(x)$** — five hard-coded Gaussian bumps on top of a
  uniform baseline $m_{\text{base}}=8$. Documents why the spec's
  $\int_\Omega m=1$ normalization is dropped (AM–GM forces $\langle\log m\rangle\le 0$,
  destroying the agent's incentive to evacuate).
* **§5 weak form / lagged linearization** — derives the backward
  implicit-Euler weak form and the lag $|\nabla v^n|^2 \to \nabla v^{n+1}\!\cdot\!\nabla v^n$.
* **§6 HJB solver** — `solve_tracer_hjb(model, m_field; T, N, σ_art, κ)`,
  one linear solve per backward timestep.
* **§7 trajectory integration** — `simulate_tracer(v_traj, x0)` does forward
  Euler on $\dot x = -\nabla v(t,x)$ with linear-interpolation exit-crossing.
* **§8 default solve and visualization** — solves with κ=1.0, σ_art=0.1,
  T=1, N=40; produces $v(t,x)$ snapshots and the trajectory-over-density plot.
* **§9 validation**:
  * **9.1** Uniform-density sanity check: trajectory vs. analytical
    straight-line travel time $\tau = L/\sqrt{2c_0}$.
  * **9.2** Optimality identity: $v(0,x_0) = J(\alpha^\ast; x_0)$ along the
    actual trajectory by trapezoidal quadrature.
* **§10 discussion** — sensitivity of $\tau$ to $\sigma_{\text{art}}$ and
  $\kappa$, limitations of forward Euler, and the explicit recipe for
  extending the solver to time-dependent $m(t,x)$ via the `_density_at`
  indirection.

### Parameter deviations from the original spec, with reasoning

| param | spec | used | reason |
|---:|---:|---:|:---|
| $\sigma_{\text{art}}$ | $10^{-3}$ | $0.1$ | Central-Galerkin without stabilization is unstable when cell Péclet $\gg 1$. With $h=0.05$ and $\vert\nabla v\vert\sim O(1)$, stability requires $\sigma_{\text{art}}^2 \gtrsim h\,\vert\nabla v\vert$. See §2.3 of the notebook. |
| $\kappa$ | $0.1$ | $1.0$ | At $\kappa=0.1$ the running cost $F$ is too small for the agent to prefer reaching the exit over lingering. See §2.3 well-posedness derivation. |
| $\int_\Omega m=1$ normalization | required | dropped | Combined with $\kappa=0.1$ this forces $\langle F\rangle\le 0$ via AM–GM. We use a uniform baseline $m_{\text{base}}=8$ instead. See §4. |

The functional form $F = \kappa\log(m+\varepsilon)$, the geometry, the mesh,
the boundary conditions, the time discretization, the lagging trick, and the
output format all match the spec literally.

### Generated artifacts (under `outputs/`)

| File | Description |
| --- | --- |
| `tracer_room.msh` | Gmsh-generated mesh |
| `tracer_mesh.png` | Triangulation with exit highlighted |
| `tracer_density.png` | Heatmap of $m(x)$ |
| `tracer_v_snapshots.png` | $v(t,x)$ at $t\in\{0, T/3, 2T/3, T\}$ |
| `tracer_trajectory.png` | Agent trajectory over $m$, with $\tau$ in the title |
| `tracer_validation_uniform.png` | Side-by-side: uniform vs. non-uniform density |

### Expected run time for `tracer_agent.ipynb`

After the MFG notebook has run once (so Gridap/Makie are precompiled), the
tracer notebook runs end-to-end in **~30 seconds** on a modern laptop CPU.
On a cold cache, allow an additional 2–3 minutes for first-time precompile.

## Package versions (pinned in `Project.toml` / `Manifest.toml`)

* `Gridap` 0.20.x
* `GridapGmsh` 0.7.x
* `Gmsh` 0.3.x
* `CairoMakie` 0.15.x (uses `record` for MP4 output)
* `IJulia` 1.34.x
* `JSON` 1.5.x (used only by the notebook builder; not by the solver)

If you hit version-resolution issues, `Pkg.instantiate()` from the
project's `Manifest.toml` should give a reproducible environment.

## Expected run time

On a modern laptop CPU, the full notebook executes end-to-end in roughly
**5–10 minutes**, dominated by:

* **First-time Gridap/Makie precompilation** (~2–3 min, one-off; cached afterwards).
* **Default Picard solve**, ~15 iterations (~30–60 s after warmup).
* **σ=0 solve** (~30–60 s).
* **MP4 rendering** for three animations (~1–2 min total at 15 fps, ~40 frames each).

If you only want the solver and static plots, skip §9.2–§10.

## Customizing parameters

All knobs live as `const` near the top of §6:

```julia
const σ_default = 0.25      # diffusion coefficient
const σ_art     = 0.05      # artificial viscosity for σ=0 case
const κ         = 0.10      # congestion strength
const ε_log     = 1e-3      # log regularization
const T_final   = 1.0
const Nt_steps  = 40
const Picard_K  = 15
const Picard_θ  = 0.3       # damping (lower → more stable, slower)
const m0_center = (0.20, 0.50)
const m0_width  = 0.10
```

If you push `σ` below ≈0.15 with the default mesh, the central-Galerkin
FP step becomes oscillatory (cell Péclet > 1). To use smaller σ, refine
the mesh (`build_default_mesh(...; h=0.02)`) or add SUPG/upwind
stabilization — see §12 of the notebook.

## Troubleshooting

* **`Gmsh` errors on first run.** `Gmsh.jl` ships its own binary; if it
  fails to load on Linux, install `libxrender1`/`libgl1`. On Windows,
  the binary should "just work" out of the box.
* **`CairoMakie` MP4 fails.** `CairoMakie.record(...; format="mp4")`
  requires `FFMPEG_jll`, which is pulled in automatically. If the
  resulting file is empty, switch to `.gif` by changing the file
  extension in §9.
* **Kernel name mismatch.** The notebook expects a kernel named
  `julia-1.12`. If yours is named differently (e.g. `julia-1.10`),
  either change the metadata in the `.ipynb` or run with
  `--ExecutePreprocessor.kernel_name=julia-1.10`.
