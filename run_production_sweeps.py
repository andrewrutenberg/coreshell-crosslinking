"""
run_production_sweeps.py -- regenerate the data behind the paper's figures.

    python3 run_production_sweeps.py                        # everything -> results/<today>/
    python3 run_production_sweeps.py --only barrier --out results/2026-08-15   # redo one part
    python3 plot_results.py results/<today>                 # then plot it

Writes into <out>/data/:
  fig1_*.npz              four single-case profile runs (Fig 1)
  phase_diagram_data.npz  60x60 (alpha, u0) sweep, bare fibril          (Figs 2, 3, 4; reference in Fig 5)
  phase_barrier.npz       the same sweep with the coat, delta_b/R = 0.1, D_b/D = 0.02 (Fig 5)

The full run takes about 1.5-2.5 h on 8 worker processes with numba.  On macOS, to leave
it running unattended:
    nohup caffeinate -i nice -n 10 python3 -u run_production_sweeps.py > sweep.log 2>&1 &
"""
import os
# One BLAS/OpenMP thread per worker process: the sweep already runs one process per core.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
from datetime import datetime

import numpy as np
import coreshell as cs

HERE = os.path.dirname(os.path.abspath(__file__))

ALPHAS = np.logspace(-2, 3, 60)
U0S = np.logspace(-3, 3, 60)
DR = 0.01            # one grid spacing for every run
DELTA_B = 0.10       # coat thickness delta_b / R: ten cells at DR
DB_RATIO = 0.02      # coat diffusivity D_b / D, so beta = DELTA_B / DB_RATIO = 5
FIG1_CASES = [       # (alpha, u0, t_end, name) for Fig 1 A-D
    (0.1,   0.1,    250.0, 'uniform'),
    (100.0, 0.1,      3.0, 'core_shell'),
    (0.1,   1000.0,   0.1, 'large_u0'),
    (0.1,   58.0,     0.5, 'kmax_0.5_boundary_u58'),   # on the K_max = 0.5 boundary
]


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main(only, out, workers):
    do = {'fig1', 'free', 'barrier'} if only == 'all' else {only}
    data_dir = os.path.join(out, 'data')
    os.makedirs(data_dir, exist_ok=True)
    t_start = time.perf_counter()
    log(f"output -> {data_dir}   ({'numba' if cs.HAVE_NUMBA else 'no numba: expect a very long run'})")

    if 'fig1' in do:
        for a, u, t_end, name in FIG1_CASES:
            res = cs.simulate(a, u, dr=DR, t_end=t_end, lin_times=True)
            cs.save_simulation_data(os.path.join(data_dir, f"fig1_{name}.npz"), res)
        log("Fig 1 profiles done")

    Kmax = Kb = None
    if 'free' in do:
        t0 = time.perf_counter()
        log(f"bare-fibril sweep, {len(ALPHAS)}x{len(U0S)} ...")
        Kmax, T50, TauMax, Peaked = cs.compute_phase_diagrams_parallel(
            ALPHAS, U0S, dr=DR, workers=workers)
        np.savez_compressed(os.path.join(data_dir, "phase_diagram_data.npz"),
                            alphas=ALPHAS, u0s=U0S, Kmax=Kmax, T50=T50,
                            TauMax=TauMax, Peaked=Peaked, dr=DR)
        log(f"bare-fibril sweep done in {(time.perf_counter() - t0) / 60:.1f} min")

    if 'barrier' in do:
        n_cells = DELTA_B / DR
        assert abs(n_cells - round(n_cells)) < 1e-9, "DELTA_B must be a whole number of grid cells"
        t0 = time.perf_counter()
        log(f"coated sweep, delta_b/R = {DELTA_B}, D_b/D = {DB_RATIO} (beta = {DELTA_B / DB_RATIO:g}) ...")
        Kb, Tb, TauMaxB, PeakedB = cs.compute_phase_diagrams_parallel(
            ALPHAS, U0S, delta=DELTA_B, Db_ratio=DB_RATIO, dr=DR, workers=workers)
        np.savez_compressed(os.path.join(data_dir, "phase_barrier.npz"),
                            alphas=ALPHAS, u0s=U0S, Kmax=Kb, T50=Tb,
                            TauMax=TauMaxB, Peaked=PeakedB,
                            delta=DELTA_B, Db_ratio=DB_RATIO, dr=DR)
        log(f"coated sweep done in {(time.perf_counter() - t0) / 60:.1f} min")

    if Kmax is not None and Kb is not None:
        j = int(np.argmin(np.abs(ALPHAS - 0.1)))
        a_free = 10 ** np.interp(0.5, Kmax[:, 0], np.log10(ALPHAS))
        a_coat = 10 ** np.interp(0.5, Kb[:, 0], np.log10(ALPHAS))
        u_free = 10 ** np.interp(0.5, Kmax[j, :], np.log10(U0S))
        u_coat = 10 ** np.interp(0.5, Kb[j, :], np.log10(U0S))
        log(f"K_max = 0.5 at u0 = {U0S[0]:.0e}: alpha = {a_free:.2f} bare, {a_coat:.2f} coated")
        log(f"K_max = 0.5 at alpha = {ALPHAS[j]:.3f}: u0 = {u_free:.1f} bare, {u_coat:.1f} coated, "
            f"shift {u_coat / u_free:.2f} (1 + 2 beta = {1 + 2 * DELTA_B / DB_RATIO:g})")

    log(f"all done in {(time.perf_counter() - t_start) / 60:.1f} min.  "
        f"Plot with:  python3 plot_results.py {os.path.relpath(out, HERE) if out.startswith(HERE) else out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', choices=['fig1', 'free', 'barrier'], default='all',
                    help='regenerate just this part (default: everything)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', datetime.now().strftime('%Y-%m-%d')),
                    help='results folder (default: results/<today>)')
    ap.add_argument('--workers', type=int, default=None,
                    help='worker processes (default: all cores but two)')
    args = ap.parse_args()
    main(args.only, os.path.abspath(args.out), args.workers)
