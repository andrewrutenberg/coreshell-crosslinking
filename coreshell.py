"""
coreshell.py  --  reaction-diffusion cross-linking in a cylindrical fibril
=========================================================================

Solves, for the mobile agent u = s/x_max and bound fraction v = x/x_max on
rho = r/R with tau = t D / R^2,

    d(u)/d(tau) = grad^2 u - alpha u (1-v),      d(v)/d(tau) = alpha u (1-v),

with u(1,tau) = u0 stepped on at tau = 0+, u(rho,0) = v(rho,0) = 0, and
no flux at the origin.  alpha = k x_max R^2 / D is the Damkohler number.

Observables (paper convention):
    A(tau)   = 2 \\int_0^1 rho v drho          -> A_max = 1
    K(tau)   = v(1,tau) - v(0,tau),  K_max = max_tau K
    tau_50%  = inf{ tau : A >= 1/2 }

Numerical scheme
----------------
Explicit forward-time centred-space on a uniform radial grid, in flux form
    grad^2 u |_i = [ F_{i+1/2} - F_{i-1/2} ] / (rho_i d_rho),
    F_{i+1/2}   = rho_{i+1/2} D_{i+1/2} (u_{i+1} - u_i) / d_rho,
with harmonic-mean face diffusivities D_{i+1/2}, so the scheme stays
conservative across the diffusivity jump at the barrier (delta > 0).  At the
origin the regular cylindrical limit grad^2 u -> 2 d2u/drho2 gives the
coefficient 4: grad^2 u|_0 = 4 D_0 (u_1 - u_0)/d_rho^2.

The time step takes the smaller of the diffusive limit 0.2 d_rho^2 (<= the
d_rho^2/4 needed for positivity at the origin node) and a reaction limit.  At
these steps u >= 0 and v <= 1 are preserved by the update itself; the clip in
the inner loop is an inactive safeguard and has been verified never to fire.

Sweeps stop once tau_50% has been passed and K has fallen 1% below its running
peak (K is single-peaked in tau), which leaves K_max and tau_50% unchanged.

checks.py exercises the solver against analytic limits; run_production_sweeps.py
produces the data behind the paper's figures.
"""

import numpy as np
from multiprocessing import Pool, cpu_count

# numba is optional: it compiles the time-stepping loop (much faster); without it
# the same loop runs as plain Python.
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(func):
        return func

# ---------------------------------------------------------------------------
# Diffusivity array builder
# ---------------------------------------------------------------------------

def _make_diffusivity(rho, Db_ratio):
    """Per-node diffusivity: 1 inside the fibril (rho <= 1), Db_ratio in the coat."""
    return np.where(rho <= 1.0 + 1e-12, 1.0, Db_ratio)


# ---------------------------------------------------------------------------
# Time-stepping loop
# ---------------------------------------------------------------------------

@njit
def _run_time_stepping_core(u, v, D_loc, rho, dr, dt, alpha, u0, i_fib, t_saves, N_r, save_profiles):
    """
    Advance u, v with explicit steps of size dt, recording A, K (and profiles)
    at the times t_saves.  Compiled by numba when it is available.
    """
    # Precompute geometric and face metrics to avoid repeating allocations
    inv_dr = 1.0 / dr
    inv_dr2 = 1.0 / (dr * dr)

    # Pre-allocate temporary workspace arrays for derivatives
    out_lap = np.zeros(N_r)
    Df = 2.0 * D_loc[:-1] * D_loc[1:] / (D_loc[:-1] + D_loc[1:])
    rho_f = (np.arange(N_r - 1) + 0.5) * dr
    F = np.zeros(N_r - 1)

    # Tracking variables
    t = 0.0
    save_idx = 0
    max_saves = len(t_saves)

    # Internal list equivalents for primitive values
    saved_ts = []
    saved_A_integrands = []
    saved_Ks = []

    # Allocate dynamic profile savers safely inside JIT types
    # Since JIT lists require matching shapes, profiles are handled structurally
    u_profiles = []
    v_profiles = []

    A_max = 1.0   # paper convention: A = 2 int_0^1 rho v drho, so A_max = 1
    K_peak = -1.0e30
    n_clip = 0    # updates that left u < 0 or v outside [0, 1] beyond roundoff

    while t <= t_saves[-1] + 0.5 * dt:

        # Check if current time matches next target save timestamp
        if save_idx < max_saves and t >= t_saves[save_idx] - 1e-14:

            # Simple trapezoidal integration for A over fibril domain (much faster inside JIT)
            # A = 2 * ∫ (ρ * v) dρ   [paper Eq. (6); A_max = 1]
            integral = 0.0
            for i in range(i_fib):
                r1, r2 = rho[i], rho[i+1]
                v1, v2 = v[i], v[i+1]
                integral += 0.5 * dr * (r1 * v1 + r2 * v2)
            A_now = 2.0 * integral
            K_now = v[i_fib] - v[0]

            saved_ts.append(t)
            saved_A_integrands.append(A_now)
            saved_Ks.append(K_now)

            # We copy snapshots safely
            u_profiles.append(u[:i_fib + 1].copy())
            v_profiles.append(v[:i_fib + 1].copy())

            save_idx += 1

            # Early exit (sweeps only): K(tau) is single-peaked, so once tau_50 has
            # been crossed and K has fallen 1% below its running peak, both K_max and
            # tau_50 are settled.  A > 0.99 covers runs whose K never turns over.
            # Remaining save slots are padded with the final state.
            if K_now > K_peak:
                K_peak = K_now
            past_peak = (K_peak > 0.0) and (K_now < 0.99 * K_peak) and (A_now >= 0.5 * A_max)

            if not save_profiles and (past_peak or A_now > 0.99 * A_max):
                while save_idx < max_saves:
                    saved_ts.append(t)
                    saved_A_integrands.append(A_now)
                    saved_Ks.append(K_now)
                    u_profiles.append(u[:i_fib + 1].copy())
                    v_profiles.append(v[:i_fib + 1].copy())
                    save_idx += 1
                break

        if save_idx >= max_saves:
            break

        # Compute Divergence Operator (Inlined & memory-stable)
        for i in range(N_r - 1):
            F[i] = rho_f[i] * Df[i] * (u[i+1] - u[i]) * inv_dr

        for i in range(1, N_r - 1):
            out_lap[i] = (F[i] - F[i-1]) / (rho[i] * dr)

        # Origin: grad^2 u -> 2 d2u/drho2, and symmetry gives d2u/drho2 = 2(u1-u0)/dr^2,
        # hence the coefficient 4.
        out_lap[0] = 4.0 * D_loc[0] * (u[1] - u[0]) * inv_dr2
        out_lap[-1] = 0.0

        # In-place Chemistry & Reaction Updates
        for i in range(i_fib + 1):
            rxn_flux = dt * alpha * u[i] * (1.0 - v[i])
            u[i] -= rxn_flux
            v[i] += rxn_flux

        # Add spatial diffusion updates globally
        for i in range(N_r):
            u[i] += dt * out_lap[i]

        # Rigid boundary clamp controls
        u[-1] = u0

        # Safeguard: within the step-size limits the update itself keeps u >= 0 and
        # 0 <= v <= 1, so any clip beyond roundoff is counted and reported.
        for i in range(N_r):
            if u[i] < 0.0:
                if u[i] < -1e-12: n_clip += 1
                u[i] = 0.0
            if v[i] < 0.0:
                if v[i] < -1e-12: n_clip += 1
                v[i] = 0.0
            elif v[i] > 1.0:
                if v[i] > 1.0 + 1e-12: n_clip += 1
                v[i] = 1.0

        t += dt

    return saved_ts, saved_A_integrands, saved_Ks, u_profiles, v_profiles, n_clip


# ---------------------------------------------------------------------------
# Core numerical solver
# ---------------------------------------------------------------------------

def simulate(alpha, u0, delta=0.0, Db_ratio=1.0, dr=0.01, dt=None,
             t_end=200.0, n_saves=2000, save_profiles=True, lin_times=False):
    """
    Run one case.

    alpha, u0        Damkohler number and scaled bath concentration
    delta, Db_ratio  coat thickness delta_b/R and diffusivity D_b/D (delta = 0: bare fibril)
    dr, dt           grid spacing and time step (dt=None: chosen for stability, see below)
    t_end, n_saves   run length and number of stored samples (log-spaced unless lin_times);
                     2000 samples resolve tau_max, the location of the contrast peak
    save_profiles    also return u(rho), v(rho) at every sample, and run to t_end;
                     otherwise stop early once K_max and tau_50 are settled

    Returns a dict with times, A, K, K_max, tau_Kmax, tau_50 (inf if A never
    reached 1/2 in the run), reached_half, t_stop, delta_eff, dr, dt, n_saves,
    n_clipped and, with save_profiles, rho_fibril, u_out, v_out.  K_max is the
    maximum over the simulated window; check reached_half before treating it as
    the maximum over the whole evolution.
    """
    for name, val in (('alpha', alpha), ('u0', u0), ('delta', delta)):
        if not (np.isfinite(val) and val >= 0):
            raise ValueError(f"{name} must be finite and >= 0, got {val}")
    if not (dr > 0 and t_end > 0 and int(n_saves) >= 2):
        raise ValueError("need dr > 0, t_end > 0 and n_saves >= 2")
    if delta > 0 and not (np.isfinite(Db_ratio) and Db_ratio > 0):
        raise ValueError(f"Db_ratio must be finite and > 0 when delta > 0, got {Db_ratio}")
    has_barrier = (delta > 0.0)

    # An exact integer number of cells inside the fibril, so a node sits on rho = 1
    i_fib = int(round(1.0 / dr))
    dr = 1.0 / i_fib

    # The coat occupies whole cells.  If delta/dr is not an integer the simulated
    # thickness (and so beta) differs from the one requested: warn, and report it.
    if has_barrier:
        n_barrier_steps = max(1, int(round(delta / dr)))
        n_pts = i_fib + n_barrier_steps + 1
        rho_max = 1.0 + n_barrier_steps * dr
        delta_eff = n_barrier_steps * dr
        if abs(delta_eff - delta) > 1e-12:
            import warnings
            warnings.warn(
                f"barrier thickness snapped to the grid: requested delta={delta:g}, "
                f"simulated delta={delta_eff:g} (dr={dr:g}). beta is rescaled by "
                f"{delta_eff/delta:.3f}. Choose dr so that delta/dr is an integer.",
                RuntimeWarning, stacklevel=2)
    else:
        delta_eff = 0.0
        n_pts = i_fib + 1
        rho_max = 1.0

    rho = np.linspace(0.0, rho_max, n_pts)
    dr = rho[1] - rho[0]
    N_r = len(rho)

    D_loc = _make_diffusivity(rho, Db_ratio if has_barrier else 1.0)

    D_max = max(1.0, Db_ratio) if has_barrier else 1.0
    # Step-size limits.  Diffusion: the origin node needs dt <= dr^2/(4 D).  Reaction:
    # u loses at most alpha*u*dt and v gains at most alpha*u0*dt per step, so
    # dt <= 1/(alpha*max(u0, 1)) keeps both in range.  The default sits well inside
    # both (paper Sec. II); an explicit dt beyond either limit is refused.
    rate = max(alpha * max(u0, 1.0), 1e-30)
    if dt is None:
        dt = min(0.2 * dr**2 / D_max, 0.1 / rate)
    elif not (0 < dt <= min(0.25 * dr**2 / D_max, 1.0 / rate)):
        raise ValueError(f"dt={dt:g} is outside the stability limits "
                         f"dt <= {0.25 * dr**2 / D_max:.3g} (diffusion), {1.0 / rate:.3g} (reaction)")

    u = np.zeros(N_r);  u[-1] = u0
    v = np.zeros(N_r)

    if lin_times:
        t_saves = np.linspace(0.0, t_end, n_saves)
    else:
        t_saves = np.concatenate([[0.0], np.logspace(np.log10(dt), np.log10(t_end), n_saves - 1)])

    saved_ts, saved_A, saved_Ks, u_p, v_p, n_clip = _run_time_stepping_core(
        u, v, D_loc, rho, dr, dt, alpha, u0, i_fib, t_saves, N_r, save_profiles
    )

    times = np.array(saved_ts)
    A_arr = np.array(saved_A)
    K_arr = np.array(saved_Ks)

    K_max    = float(np.max(K_arr)) if len(K_arr) > 0 else 0.0
    tau_Kmax = float(times[np.argmax(K_arr)]) if len(K_arr) > 0 else 0.0
    half = 0.5           # A_max = 1 (paper convention), so 50% is A = 0.5
    idx50 = np.searchsorted(A_arr, half)

    if idx50 == 0:
        tau_50 = float(times[0])
    elif idx50 < len(times):
        # linear interpolation between the samples either side of A = 1/2
        t0, t1 = times[idx50 - 1], times[idx50]
        A0, A1 = A_arr[idx50 - 1], A_arr[idx50]
        tau_50 = float(t0 + (half - A0) * (t1 - t0) / (A1 - A0))
    else:
        tau_50 = float('inf')

    if n_clip:
        import warnings
        warnings.warn(f"{n_clip} updates were clipped (alpha={alpha:g}, u0={u0:g}, dr={dr:g}, "
                      f"dt={dt:g}); the result is not reliable", RuntimeWarning, stacklevel=2)

    result = dict(times=times, A=A_arr, K=K_arr,
                  K_max=K_max, tau_Kmax=tau_Kmax, tau_50=tau_50,
                  reached_half=bool(np.isfinite(tau_50)), t_stop=float(times[-1]),
                  alpha=alpha, u0=u0, delta=delta, delta_eff=delta_eff, Db_ratio=Db_ratio,
                  dr=float(dr), dt=float(dt), n_saves=int(n_saves), n_clipped=int(n_clip))

    if save_profiles:
        result['rho_fibril'] = rho[:i_fib + 1]
        result['u_out']      = np.array(u_p)
        result['v_out']      = np.array(v_p)

    return result


def refine_peak_time(times, K, tol=0.01):
    """
    Locate the maximum of K(tau) to sub-save-step accuracy.  Returns
    (tau_max, peaked).

    The save grid is logarithmic, so the parabola is fitted in log10(tau): that
    is where the peak is locally symmetric, and it stays well conditioned across
    the four decades the grid spans.  Taking the nearest save point instead
    quantises tau_max in ~2% steps, which terraces the contours.

    peaked is False when K has not turned over by the end of the run.  The
    stepper pads its remaining saves with the last value on early exit, so
    argmax then lands on the stopping time -- a numerical artefact, not a
    physical peak.  Callers should mask those points.  (Over the sweep grid this
    does not currently fire: K turns over everywhere.)
    """
    if len(K) == 0:
        return float('nan'), False

    i = int(np.argmax(K))
    K_peak = float(K[i])

    # Has contrast actually turned over?  On a front-limited plateau K is still
    # at its maximum when the run stops, so this test fails, as intended.
    peaked = K_peak > 0.0 and float(K[-1]) < (1.0 - tol) * K_peak
    if not peaked:
        return float(times[i]), False

    # Turning over is necessary but not sufficient.  Where the contrast saturates
    # (K_max = 1 to machine precision: v(1) = 1 and v(0) = 0), K has a FLAT TOP
    # rather than a peak -- hundreds of samples sit within 1e-12 of the maximum,
    # spanning up to a factor of 300 in tau.  argmax then returns the first sample
    # to round up to 1.0, which lands at alpha*u0*tau ~ 33.3 because that is where
    # 1 - exp(-alpha u0 tau) saturates in double precision.  That is a
    # floating-point onset time, not a peak location, and it affects ~14% of the
    # plotted plane.  K still decays later (the core fills), so the turnover test
    # above passes; the maximum must additionally be localised.
    if len(times) > 2:
        near = np.where(K >= K_peak * (1.0 - 1e-9))[0]
        if len(near) > 1:
            t_lo, t_hi = float(times[near[0]]), float(times[near[-1]])
            if t_lo > 0.0 and np.log10(t_hi / t_lo) > 0.05:   # decades of plateau
                return float(times[i]), False

    t_peak = float(times[i])

    # Needs an interior point with two strictly separated neighbours in log
    # space.  times[0] is exactly 0, and the padded tail repeats one timestamp,
    # so both ends are skipped.
    if i == 0 or i >= len(times) - 1:
        return t_peak, True
    t0, t1, t2 = float(times[i - 1]), float(times[i]), float(times[i + 1])
    if not (0.0 < t0 < t1 < t2):
        return t_peak, True

    x = np.log10([t0, t1, t2])
    y = np.array([K[i - 1], K[i], K[i + 1]], dtype=float)

    a, b, _ = np.polyfit(x, y, 2)     # exact quadratic through three points
    if a >= 0.0:                      # not concave -> no interior maximum
        return t_peak, True
    x_peak = -b / (2.0 * a)
    if not (x[0] <= x_peak <= x[2]):  # vertex outside the bracket -> distrust
        return t_peak, True

    return float(10.0 ** x_peak), True


def _worker_simulate(args):
    """One sweep point: K_max, tau_50, tau_max, whether K has a true peak, clip count."""
    a, u, delta, Db_ratio, dr, dt = args
    r = simulate(a, u, delta=delta, Db_ratio=Db_ratio, dt=dt, dr=dr, t_end=100000, save_profiles=False)
    tau_max, peaked = refine_peak_time(r['times'], r['K'])
    return r['K_max'], r['tau_50'], tau_max, peaked, r['n_clipped']

# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def compute_phase_diagrams_parallel(alphas, u0s, delta=0.0, Db_ratio=1.0, dr=0.01, dt=None,
                                    workers=None, verbose=True):
    """
    Run every (alpha, u0) pair in parallel.  Returns (Kmax, T50, TauMax, Peaked), each
    shaped (len(alphas), len(u0s)).  Peaked is False where TauMax is not a true peak
    time -- K never turned over, or the contrast saturates (K_max = 1, a plateau) --
    and TauMax must be masked there.  workers defaults to all cores but two.
    """
    workers = workers or max(1, cpu_count() - 2)
    if verbose:
        print(f"Sweeping {len(alphas)}x{len(u0s)} grid on {workers} worker processes "
              f"({'numba' if HAVE_NUMBA else 'no numba: plain Python, much slower'})...")

    tasks = []
    for a in alphas:
        for j in u0s:
            tasks.append((a, j, delta, Db_ratio, dr, dt))

    # chunksize=1: the cost per point spans ~4 decades (the small-alpha*u0 corner runs
    # the full window), so the default chunking hands the long poles to one worker.
    with Pool(processes=workers) as pool:
        results = pool.map(_worker_simulate, tasks, chunksize=1)

    shape  = (len(alphas), len(u0s))
    Kmax   = np.empty(shape)
    T50    = np.empty(shape)
    TauMax = np.empty(shape)
    Peaked = np.empty(shape, dtype=bool)

    idx, n_clipped_total = 0, 0
    for i in range(len(alphas)):
        for j in range(len(u0s)):
            Kmax[i, j], T50[i, j], TauMax[i, j], Peaked[i, j], n_clip = results[idx]
            n_clipped_total += n_clip
            idx += 1
    Peaked &= Kmax < 1.0 - 1e-9
    if n_clipped_total:
        print(f"WARNING: {n_clipped_total} clipped updates across the sweep; results unreliable")

    return Kmax, T50, TauMax, Peaked


def save_simulation_data(filename, data_dict):
    """Write the arrays and scalars of a simulate() result to a compressed .npz."""
    save_kwargs = {}
    for key, value in data_dict.items():
        if isinstance(value, (np.ndarray, list)):
            save_kwargs[key] = np.array(value)
        elif isinstance(value, (int, float, str, bool)):
            save_kwargs[key] = value

    np.savez_compressed(filename, **save_kwargs)
    print(f"saved {filename}")
