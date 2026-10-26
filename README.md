# Reaction-diffusion cross-linking of a cylindrical fiber

Code and data for

> X. Gouws, U. Rajnis, A. Mastnak, L. Kreplak and A. D. Rutenberg,
> *Diffusion-limited cross-linking of biopolymer fibers: core-shell versus uniform profiles*
> (submitted to *Soft Matter*, 2026).

A diffusible agent enters an infinite cylinder from a maintained bath and attaches
irreversibly to a finite density of reactive sites. In scaled variables
(ρ = r/R, τ = tD/R², u = s/x_max, v = x/x_max)

    ∂u/∂τ = ∇²u − α u (1 − v),      ∂v/∂τ = α u (1 − v),

with u(1,τ) = u₀ for τ > 0, no flux at ρ = 0, and u = v = 0 initially. The control
parameters are the Damköhler number α = k x_max R²/D and the scaled bath concentration
u₀ = s₀/x_max. The observables are the bound fraction A(τ) = 2∫₀¹ρ v dρ (A_max = 1), the
radial contrast K(τ) = v(1,τ) − v(0,τ) with its maximum K_max and peak time τ_max, and the
half-conversion time τ₅₀%. An optional non-reacting coat of thickness δ_b/R and
diffusivity D_b/D models a proteoglycan barrier.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 plot_results.py results/2026-08-15   # the paper's figures from the supplied data (seconds)
python3 checks.py                            # solver against analytic limits (~15 s)
```

One case, from Python:

```python
from coreshell import simulate
r = simulate(alpha=100, u0=0.1, t_end=3.0)       # Fig. 1B
print(r['K_max'], r['tau_50'], r['reached_half'])
```

Regenerate all the data (about 1.5–2.5 h on 8 worker processes with numba), then plot it:

```bash
python3 run_production_sweeps.py                 # writes results/<today>/
python3 plot_results.py results/<today>
```

`run_production_sweeps.py --only fig1|free|barrier` redoes one part, `--out` picks the
folder, `--workers` sets the number of processes (default: all cores but two).
`plot_results.py` with no argument plots the newest folder in `results/`, and stops with an
error if any input file is missing.

## Files

| file | role |
|---|---|
| `coreshell.py` | the solver: `simulate()` for one case, `compute_phase_diagrams_parallel()` for a grid. Its docstrings document the scheme. Needs numpy only (numba optional). |
| `run_production_sweeps.py` | writes every data file the figures need. Parameters at the top. |
| `plot_results.py` | Fig1–Fig5 (`.pdf` and `.png`) from one results folder. |
| `checks.py` | ten pass/fail comparisons with closed-form results: the five limits of the paper's Sec. II (surface solution, uniform regime, early-time τ^{3/2}, late-time rate, the two K_max asymptotes), global mass balance, the sharp-front τ₅₀%, and the effect of the coat in the reaction- and front-limited limits. Exit status 1 on any failure. |
| `results/2026-08-15/` | the data, figures and run log behind the paper. |

## Numerical scheme

Explicit forward-time, centred-space on a uniform radial grid, Δρ = 0.01 for every run, in
conservative flux form with harmonic-mean face diffusivities (needed where D jumps at the coat).
At the origin, ∇²u → 2∂²u/∂ρ², i.e. 4(u₁ − u₀)/Δρ². The default time step is
min(0.2 Δρ²/D_max, 0.1/[α max(u₀, 1)]); `simulate()` refuses an explicit step beyond the
stability limits Δρ²/(4 D_max) and 1/[α max(u₀, 1)], and rejects invalid parameters. Within
these limits u ≥ 0 and 0 ≤ v ≤ 1 hold without clipping; any clipping is counted
(`n_clipped`) and raises a warning.

A is integrated by the trapezoidal rule and τ₅₀% interpolated between samples. τ_max is refined
by a parabola in log τ through the sampled peak. Sweep runs stop once τ₅₀% has been passed and
K has fallen 1% below its running peak; this leaves K_max and τ₅₀% unchanged. A single
`simulate()` call runs to `t_end` (default 200), which is too short for the slowest
cases: `tau_50 = inf` and `reached_half = False` mean half conversion was not reached, and
K_max is then only the maximum over the simulated window.

## Data → figures

| data file | used in |
|---|---|
| `fig1_uniform.npz`, `fig1_core_shell.npz`, `fig1_large_u0.npz`, `fig1_kmax_0.5_boundary_u58.npz` | Fig. 1 A–D: (α, u₀) = (0.1, 0.1), (100, 0.1), (0.1, 1000), (0.1, 58) |
| `phase_diagram_data.npz` | Figs. 2 (`Kmax`), 3 (`T50`), 4 (`TauMax`, `Peaked`), and the reference contour in Fig. 5 |
| `phase_barrier.npz` | Fig. 5: δ_b/R = 0.10, D_b/D = 0.02 (β = 5) |
| `phase_diagram_data_dt0p4.npz`, `phase_barrier_dt0p4.npz`, `phase_diagram_data_nsaves500.npz` | not plotted: the same sweeps with a 4× larger reaction time step (0.4 instead of 0.1 in the reaction limit) and with 500 instead of 2000 stored samples. Kept as convergence evidence: within the plotted windows K_max changes by < 0.001, τ₅₀% by < 0.3% and τ_max by < 0.1%. These two variants were made with an earlier version of the scripts. |

Sweep files hold `alphas`, `u0s` and 60 × 60 arrays indexed `[alpha, u0]`, plus the run
parameters (`dr`; for the coat, `delta` and `Db_ratio`, which `plot_results.py` reads so the
1 + 2β line in Fig. 5 always matches the sweep). `TauMax` is a peak time only where `Peaked`
is True, and must be masked elsewhere: there K never turned over, or the contrast saturates
(K_max = 1), so the maximum is a plateau. The `Peaked` arrays in this folder were corrected on
2026-10-26 to apply that saturation rule, which the August sweep predated; nothing else in the
files changed, and no figure depends on the change.

Headline numbers from this data: K_max = 0.5 at α = 11.27 for u₀ ≪ 1 (11.28 with the coat) and at
αu₀ ≈ 5.8 for u₀ ≫ 1; the coat moves the high-u₀ boundary by 9.9× against the predicted
1 + 2β = 11.

## License

MIT — see `LICENSE`.
