"""
checks.py -- the solver against analytic limits (paper Sec. II) and two coat estimates.

    python3 checks.py        # about 30 s with numba; exit status 1 if any check fails

Each check compares one simulate() run, or a pair, with a closed-form result and
passes within a stated tolerance.  None is needed to reproduce the figures.
"""
import sys
import numpy as np
from coreshell import simulate

LN2 = np.log(2.0)
C_FRONT = 0.038357   # integral of rho ln(1/rho) from 1/sqrt(2) to 1: tau_50 u0 for a sharp front


def run(*args, **kw):
    return simulate(*args, save_profiles=False, **kw)


def trapezoid(y, x):
    return 0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x))


def check_surface():
    """v(1, tau) = 1 - exp(-alpha u0 tau) exactly: the surface sees the bath at all times."""
    r = simulate(1.0, 1.0, t_end=2.0, n_saves=200)
    err = np.max(np.abs(r['v_out'][:, -1] - (1 - np.exp(-r['times']))))
    return err < 1e-4, f"max |v(1) - exact| = {err:.1e}  (tol 1e-4)"


def check_uniform():
    """alpha << 1, alpha u0 << 1: A(tau) = 1 - exp(-alpha u0 tau) after the filling transient."""
    a, u0 = 0.01, 0.1
    r = run(a, u0, dr=0.02, t_end=3000, n_saves=400)
    m = r['times'] > 50
    err = np.max(np.abs(r['A'][m] - (1 - np.exp(-a * u0 * r['times'][m]))))
    return err < 1e-3, f"max |A - exact| = {err:.1e}  (tol 1e-3)"


def check_early_time():
    """A ~ (8/(3 sqrt pi)) alpha u0 tau^{3/2} - (1/2) alpha u0 tau^2 at early times (paper Eq. 10)."""
    a, u0 = 1.0, 0.01
    r = simulate(a, u0, dr=0.0005, t_end=1e-4, n_saves=50, lin_times=True)
    t = r['times'][-1]
    ratio = r['A'][-1] / ((8 / (3 * np.sqrt(np.pi))) * a * u0 * t**1.5 - 0.5 * a * u0 * t**2)
    return abs(ratio - 1) < 5e-3, f"A / asymptote at tau = 1e-4: {ratio:.5f}  (tol 0.5%)"


def check_late_rate():
    """1 - A decays at rate alpha u0 at late times (paper Eq. 13)."""
    a, u0 = 1.0, 1.0
    r = simulate(a, u0, t_end=15.0, n_saves=600)
    m = (r['times'] > 5) & (r['times'] < 12)
    rate = -np.polyfit(r['times'][m], np.log(1 - r['A'][m]), 1)[0] / (a * u0)
    return abs(rate - 1) < 5e-3, f"fitted rate / (alpha u0) = {rate:.5f}  (tol 0.5%)"


def check_kmax_small_u0():
    """K_max ~ alpha/16 for alpha << 1, u0 << 1 (paper Eq. 8)."""
    a = 0.1
    r = run(a, 0.001, dr=0.02, t_end=40000)
    ratio = r['K_max'] / (a / 16)
    return r['reached_half'] and abs(ratio - 1) < 0.02, f"K_max / (alpha/16) = {ratio:.4f}  (tol 2%)"


def check_kmax_large_u0():
    """K_max ~ alpha u0/4 for u0 >> 1, alpha u0 << 1 (paper Eq. 9)."""
    a, u0 = 1e-4, 100.0
    r = run(a, u0, t_end=1000)
    ratio = r['K_max'] / (a * u0 / 4)
    return r['reached_half'] and abs(ratio - 1) < 0.03, f"K_max / (alpha u0/4) = {ratio:.4f}  (tol 3%)"


def check_global_balance():
    """d/dtau [2 int rho (u + v) drho] equals the flux 2 du/drho through rho = 1."""
    dr = 0.01
    r = simulate(1.0, 0.1, dr=dr, t_end=10.0, lin_times=True)
    rho, u, v = r['rho_fibril'], r['u_out'], r['v_out']
    M = np.array([2 * trapezoid(rho * (u[k] + v[k]), rho) for k in range(len(r['times']))])
    lhs = np.gradient(M, r['times'])
    rhs = 2 * (1 - 0.5 * dr) * (u[:, -1] - u[:, -2]) / dr
    err = np.max((np.abs(lhs - rhs) / (np.abs(rhs) + 1e-12))[5:-2])   # skip the start-up step and the end
    return err < 0.05, f"max relative mismatch = {err:.2%}  (tol 5%)"


def check_front():
    """Large alpha, small u0: a sharp front gives tau_50 u0 -> 0.0384 (paper Eq. 11)."""
    ratios = [run(1000.0, u0, t_end=20)['tau_50'] * u0 / C_FRONT for u0 in (0.01, 0.03, 0.1)]
    ok = all(1.0 <= q < 1.10 for q in ratios)
    return ok, "tau_50 u0 / 0.0384 at u0 = 0.01, 0.03, 0.1: " + ", ".join(f"{q:.3f}" for q in ratios) + \
        "  (tol 0-10%; alpha = 1000 is not yet a sharp front)"


def check_coat_small_alpha():
    """Coat, reaction-limited: tau_50 slows by 1 + alpha beta/(4 ln 2).

    Uniform uptake through the coat's series resistance gives dv/dtau =
    alpha u0 (1-v) / (1 + alpha beta (1-v)/2) (paper Eq. 15 with the site depletion kept);
    integrating to v = 1/2 gives the factor above.
    """
    a, u0, delta, db = 0.01, 0.1, 0.1, 0.02
    beta = delta / db
    s = run(a, u0, delta=delta, Db_ratio=db, dr=0.02, t_end=3000)['tau_50'] / \
        run(a, u0, dr=0.02, t_end=3000)['tau_50']
    pred = 1 + a * beta / (4 * LN2)
    ok = abs((s - 1) / (pred - 1) - 1) < 0.2
    return ok, f"slowdown {s:.4f}, predicted {pred:.4f}  (excess within 20%)"


def check_coat_front():
    """Coat, front-limited: tau_50 u0 = 0.0384 + beta_c/4, beta_c = (D/D_b) ln(1 + delta/R).

    The front sees the coat's resistance in series with its own, ln(1/rho_f);
    beta_c is the thick-shell form of beta.
    """
    u0, delta, db = 0.01, 0.1, 0.02
    pred = C_FRONT + np.log(1 + delta) / db / 4
    q = run(1000.0, u0, delta=delta, Db_ratio=db, t_end=1000)['tau_50'] * u0 / pred
    return abs(q - 1) < 0.05, f"tau_50 u0 / prediction = {q:.4f}  (tol 5%)"


CHECKS = [check_surface, check_uniform, check_early_time, check_late_rate,
          check_kmax_small_u0, check_kmax_large_u0, check_global_balance, check_front,
          check_coat_small_alpha, check_coat_front]

if __name__ == '__main__':
    failed = 0
    for check in CHECKS:
        ok, detail = check()
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {check.__name__:24s} {detail}", flush=True)
    print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
