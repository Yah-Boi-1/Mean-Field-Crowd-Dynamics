# Mean Field Games for Crowd Evacuation — Julia/Gridap notebook

A single-notebook implementation of a 2D **second-order Mean Field Game**
(coupled HJB ↔ Fokker–Planck) on an unstructured triangle mesh, with
extensive pedagogical commentary, agent-level simulation, and animated
visualization.

The math, derivations, and discussion live inside
[`mfg_crowd_dynamics.ipynb`](mfg_crowd_dynamics.ipynb). This README
just covers how to run it.

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
   Then open `mfg_crowd_dynamics.ipynb` and Run All.

   Alternatively, headless run from the shell:
   ```bash
   jupyter nbconvert --to notebook --execute mfg_crowd_dynamics.ipynb \
       --output mfg_crowd_dynamics.ipynb --ExecutePreprocessor.timeout=1800
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
