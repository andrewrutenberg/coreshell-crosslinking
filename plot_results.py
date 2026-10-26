"""
plot_results.py -- Figures 1-5 of the paper from one results folder.

    python3 plot_results.py                        # newest folder in results/
    python3 plot_results.py results/2026-08-15     # a specific one (the paper's data)

Reads <folder>/data/*.npz written by run_production_sweeps.py and writes
<folder>/plots/Fig1..Fig5 as .pdf and .png.  Exits with an error if any input is missing.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patheffects import withStroke

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Style (Soft Matter)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 11,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,   # embed editable TrueType text in PDF output
})

CONTOUR_LABEL_FONTSIZE = 10

# Axis and colourbar labels: regular mathtext thickened by a thin same-colour stroke, in
# points (regular -> bold is ~0.9 pt of stem at 13 pt).  0 = plain, which also keeps the
# labels as live text in the PDF; any stroke turns them into outlines.
AXIS_LABEL_STROKE_PT = 0.05

# Phase-diagram colourbars: gap to the axes and height, as fractions of the axes, and the
# label's position along the bar (raised so it does not line up with the alpha label).
CBAR_PAD = 0.08
CBAR_SHRINK = 0.85
CBAR_LABEL_Y = 0.75

# Phase-diagram windows, as log10 exponents.  Figs 2, 3, 5: u0 1e-3..1e3, alpha 1e-1..1e3.
# Fig 4 uses a smaller window: tau_max is undefined where the contrast saturates
# (K_max = 1), which is 18% of the full window but 2% of this one, drawn grey.
PHASE_XLIM = (-3, 3)
PHASE_YLIM = (-1, 3)
TAUMAX_XLIM = (-3, 2)
TAUMAX_YLIM = (-1, 2)


def heavier():
    """path_effects implementing AXIS_LABEL_STROKE_PT, for set_xlabel/set_ylabel/set_label."""
    return [withStroke(linewidth=AXIS_LABEL_STROKE_PT, foreground='black')] if AXIS_LABEL_STROKE_PT > 0 else []


def save_figure(fig, plots_dir, name):
    fig.savefig(os.path.join(plots_dir, f'{name}.pdf'))
    fig.savefig(os.path.join(plots_dir, f'{name}.png'), dpi=300)
    print(f"Saved {name}")


# ---------------------------------------------------------------------------
# Axis helpers (phase diagrams are drawn in log10 coordinates)
# ---------------------------------------------------------------------------

def set_log_axes(ax, xlim, ylim):
    """Limits, decade tick labels 10^k, and 2..9 minor ticks for log10-coordinate axes.

    Ticks are mirrored on all four sides: the contours are long and steep, and a second
    set of ticks keeps the eye reading square to the axes rather than along a contour.
    """
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    x_ticks = np.arange(int(np.ceil(xlim[0])), int(np.floor(xlim[1])) + 1)
    y_ticks = np.arange(int(np.ceil(ylim[0])), int(np.floor(ylim[1])) + 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f'$10^{{{int(k)}}}$' for k in x_ticks])
    ax.set_yticklabels([f'$10^{{{int(k)}}}$' for k in y_ticks])

    def minor(lim):
        lo, hi = int(np.floor(lim[0])), int(np.ceil(lim[1]))
        return [k + np.log10(m) for k in range(lo, hi) for m in range(2, 10)]
    ax.xaxis.set_minor_locator(FixedLocator(minor(xlim)))
    ax.yaxis.set_minor_locator(FixedLocator(minor(ylim)))
    ax.tick_params(axis='both', which='minor', length=3, width=0.6, color='black',
                   direction='out', top=True, right=True)
    ax.tick_params(axis='both', which='major', direction='out',
                   top=True, right=True, labeltop=False, labelright=False)
    ax.set_xlabel(r'$u_0$', path_effects=heavier())
    ax.set_ylabel(r'$\alpha$', path_effects=heavier())


def crop(alphas, u0s, field, xlim, ylim):
    """log10 coordinates and the sub-array of field inside the window."""
    la, lu = np.log10(alphas), np.log10(u0s)
    ua = (lu >= xlim[0]) & (lu <= xlim[1])
    aa = (la >= ylim[0]) & (la <= ylim[1])
    return lu[ua], la[aa], field[np.ix_(aa, ua)]


def decimal_label(x):
    """Contour label for a level given as log10: 1000, 10, 1, 0.1, 0.01, ..."""
    val = 10 ** x
    if val >= 1.0:
        return f'{val:.0f}'
    if val >= 1e-3:
        dec = int(np.ceil(-np.log10(val))) + 1
        return f'{val:.{dec}f}'.rstrip('0').rstrip('.')
    return f'{val:g}'


def power_label(x):
    """10^k for decade levels, else a decimal."""
    k = round(x)
    return f'$10^{{{int(k)}}}$' if abs(x - k) < 1e-9 else decimal_label(x)


# ---------------------------------------------------------------------------
# Fig 1: bound-site profiles
# ---------------------------------------------------------------------------

def plot_profiles(result, ax, times_to_plot=6):
    """v(rho) at evenly spaced times: line weight and colour (black -> blue) grow with time."""
    rho, times, v_out = result['rho_fibril'], result['times'], result['v_out']
    idxs = np.round(np.linspace(0, len(times) - 1, times_to_plot)).astype(int)

    # (linestyle, linewidth), earliest first; the first sample is tau = 0 (v = 0)
    styles = [((0, (1, 10)), 1.0), (':', 1.3), ('--', 1.6), ('-.', 1.9),
              ('-', 2.4), ('-', 3.0), ('-', 3.5)][:times_to_plot]
    # thin dark lines and thick light ones carry about the same visual weight
    line_cmap = LinearSegmentedColormap.from_list(
        'black_to_blue', ['#000000', '#14225a', '#1e4bb8', '#2f7fe8'])
    colors = line_cmap(np.linspace(0.0, 1.0, times_to_plot))

    for k, idx in enumerate(idxs):
        ls, lw = styles[k]
        ax.plot(rho, v_out[idx], color=colors[k], linestyle=ls, linewidth=lw)
    ax.set_xlabel(r'$\rho = r/R$', path_effects=heavier())
    ax.set_ylabel(r'$v(\rho,\tau)$', path_effects=heavier())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)


# ---------------------------------------------------------------------------
# Figs 2 and 5: maximum contrast K_max
# ---------------------------------------------------------------------------

def plot_Kmax_heatmap(alphas, u0s, Kmax, ax, reference=None, delta=None, Db_ratio=None,
                      xlim=PHASE_XLIM, ylim=PHASE_YLIM):
    """
    K_max over (u0, alpha) with labelled contours.

    With reference (a bare-fibril sweep: dict with alphas, u0s, Kmax) this is Fig 5: the
    coated sweep, plus the bare K_max = 0.5 contour (green dashed) and that contour
    moved to larger u0 by the factor 1 + 2*beta of paper Eq. (shift), beta =
    (delta_b/R)(D/D_b) (red dotted).
    """
    xs, ys, field = crop(alphas, u0s, Kmax, xlim, ylim)
    softer = LinearSegmentedColormap.from_list(
        'light_coolwarm', plt.get_cmap('coolwarm')(np.linspace(0.1, 0.9, 256)))
    # rasterized: one image in the PDF, without hairline seams between cells
    im = ax.pcolormesh(xs, ys, field, norm=plt.Normalize(0, 1), cmap=softer,
                       shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, pad=CBAR_PAD, shrink=CBAR_SHRINK).set_label(
        r'$K_{\max}$', path_effects=heavier(), y=CBAR_LABEL_Y)

    if reference is not None:
        rxs, rys, rfield = crop(reference['alphas'], reference['u0s'], reference['Kmax'], xlim, ylim)
        ax.contour(rxs, rys, rfield, levels=[0.5], colors='green', linestyles='dashed',
                   linewidths=3.0, alpha=0.9)
        beta = delta / Db_ratio
        ax.contour(rxs + np.log10(1.0 + 2.0 * beta), rys, rfield, levels=[0.5],
                   colors='red', linestyles='dotted', linewidths=4, alpha=0.95)
        levels, widths = [0.1, 0.5, 0.9], [1.0, 1.5, 1.0]
    else:
        levels, widths = [0.1, 0.3, 0.5, 0.7, 0.9], [1.0, 1.0, 1.75, 1.0, 1.0]
    contours = ax.contour(xs, ys, field, levels=levels, colors='black',
                          linewidths=widths, alpha=0.8)
    ax.clabel(contours, inline=True, fmt='%1.1f', fontsize=CONTOUR_LABEL_FONTSIZE, colors='black')
    set_log_axes(ax, xlim, ylim)


# ---------------------------------------------------------------------------
# Fig 3: time to half conversion
# ---------------------------------------------------------------------------

def plot_tau50_heatmap(alphas, u0s, T50, ax, levels=(0.001, 0.01, 0.1, 1, 10, 100, 1000),
                       xlim=PHASE_XLIM, ylim=PHASE_YLIM):
    """tau_50% over (u0, alpha), log colour scale, white decade contours."""
    xs, ys, field = crop(alphas, u0s, np.log10(T50), xlim, ylim)
    im = ax.pcolormesh(xs, ys, field, cmap='viridis_r', shading='auto',
                       vmin=-3.0, vmax=3.0, rasterized=True)
    cbar = plt.colorbar(im, ax=ax, pad=CBAR_PAD, shrink=CBAR_SHRINK)
    cbar.set_label(r'$\tau_{50\%}$', path_effects=heavier(), y=CBAR_LABEL_Y)
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'$10^{{{int(t)}}}$' if t.is_integer() else f'$10^{{{t:.1f}}}$' for t in ticks])

    use = [l for l in levels if np.nanmin(field) <= np.log10(l) <= np.nanmax(field)]
    if use:
        contours = ax.contour(xs, ys, field, levels=np.log10(use), colors='white',
                              linestyles='solid', linewidths=0.8)
        ax.clabel(contours, inline=True, inline_spacing=2, fmt=decimal_label,
                  fontsize=CONTOUR_LABEL_FONTSIZE, colors='white')
    set_log_axes(ax, xlim, ylim)


# ---------------------------------------------------------------------------
# Fig 4: time of peak contrast
# ---------------------------------------------------------------------------

def plot_taumax(alphas, u0s, TauMax, Peaked, Kmax, levels_B=(0.4, 0.5, 0.6, 0.8),
                xlim=TAUMAX_XLIM, ylim=TAUMAX_YLIM):
    """
    A) tau_max, the time at which K(tau) peaks, with decade contours.
    B) tau_max / sqrt(tau_s tau_tr), with tau_s = 1/(alpha u0) and tau_tr = max(1, 1/u0):
       close to 1/2 for alpha >~ 1, approaching ln2/sqrt(alpha) in the uniform regime.

    Grey where tau_max is not a peak time: Peaked is False, or the contrast saturates
    (K_max = 1, so the maximum is a plateau).
    """
    valid = Peaked & np.isfinite(TauMax) & (TauMax > 0) & (Kmax < 1.0 - 1e-9)
    a_mesh = alphas[:, None]
    u_mesh = u0s[None, :]
    collapse = TauMax / np.sqrt(1.0 / (a_mesh * u_mesh) * np.maximum(1.0, 1.0 / u_mesh))
    with np.errstate(divide='ignore', invalid='ignore'):
        fields = [np.where(valid, np.log10(np.where(valid, f, 1.0)), np.nan)
                  for f in (TauMax, collapse)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    panels = [(fields[0], r'$\tau_{\max}$', None, power_label, 'A)'),
              (fields[1], r'$\tau_{\max}/\sqrt{\tau_s \tau_{\rm tr}}$', list(levels_B), decimal_label, 'B)')]

    for ax, (full, cbar_label, levels, fmt, panel) in zip(axes.flat, panels):
        xs, ys, field = crop(alphas, u0s, full, xlim, ylim)
        finite = field[np.isfinite(field)]
        lo, hi = int(np.floor(finite.min())), int(np.ceil(finite.max()))
        ticks = np.arange(lo, hi + 1)
        if len(ticks) > 7:
            ticks = np.array(sorted({lo, hi} | {t for t in ticks if t % 2 == 0}))

        cmap = plt.get_cmap('viridis_r').copy()
        cmap.set_bad('0.86')
        im = ax.pcolormesh(xs, ys, np.ma.masked_invalid(field), vmin=lo, vmax=hi,
                           cmap=cmap, shading='auto', rasterized=True)
        cbar = plt.colorbar(im, ax=ax, pad=CBAR_PAD, shrink=CBAR_SHRINK)
        cbar.set_label(cbar_label, path_effects=heavier(), y=CBAR_LABEL_Y)
        cbar.locator = FixedLocator(ticks)
        cbar.formatter = FixedFormatter([f'$10^{{{int(t)}}}$' for t in ticks])
        cbar.update_ticks()

        if levels is None:   # decade contours inside the data range
            levels = [10.0 ** k for k in range(lo, hi + 1)]
        use = [l for l in levels if finite.min() <= np.log10(l) <= finite.max()]
        if use:
            contours = ax.contour(xs, ys, field, levels=np.log10(use), colors='white',
                                  linestyles='solid', linewidths=0.8, alpha=0.9)
            ax.clabel(contours, inline=True, inline_spacing=2, fmt=fmt,
                      fontsize=CONTOUR_LABEL_FONTSIZE, colors='white')

        set_log_axes(ax, xlim, ylim)
        ax.tick_params(axis='y', labelleft=True)   # sharey would hide panel B's alpha labels
        ax.text(-0.16, 1.0, panel, transform=ax.transAxes, fontsize=12, va='top', ha='right')

    fig.tight_layout(w_pad=3.0)   # room between panel A's colourbar and panel B
    return fig


# ---------------------------------------------------------------------------

def newest_results_folder():
    folders = sorted(d for d in glob.glob(os.path.join(HERE, 'results', '*'))
                     if os.path.isdir(os.path.join(d, 'data')))
    if not folders:
        sys.exit("No results/<date>/data folders found; run run_production_sweeps.py first.")
    return folders[-1]


def main(folder):
    data_dir = os.path.join(folder, 'data')
    plots_dir = os.path.join(folder, 'plots')
    fig1_names = ['uniform', 'core_shell', 'large_u0', 'kmax_0.5_boundary_u58']
    needed = [f'fig1_{n}.npz' for n in fig1_names] + ['phase_diagram_data.npz', 'phase_barrier.npz']
    missing = [f for f in needed if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        sys.exit(f"{data_dir}: missing {', '.join(missing)}")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plotting from {data_dir}")

    # Fig 1: (alpha, u0) = (0.1, 0.1), (100, 0.1), (0.1, 1000), (0.1, 58)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, name, panel in zip(axes.flat, fig1_names, ['A)', 'B)', 'C)', 'D)']):
        plot_profiles(np.load(os.path.join(data_dir, f'fig1_{name}.npz')), ax)
        ax.text(-0.19, 1.0, panel, transform=ax.transAxes, fontsize=12, ha='left', va='top')
    fig.tight_layout()
    save_figure(fig, plots_dir, 'Fig1')

    free = np.load(os.path.join(data_dir, 'phase_diagram_data.npz'))
    alphas, u0s, Kmax = free['alphas'], free['u0s'], free['Kmax']

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_Kmax_heatmap(alphas, u0s, Kmax, ax)
    fig.tight_layout()
    save_figure(fig, plots_dir, 'Fig2')

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_tau50_heatmap(alphas, u0s, free['T50'], ax)
    fig.tight_layout()
    save_figure(fig, plots_dir, 'Fig3')

    save_figure(plot_taumax(alphas, u0s, free['TauMax'], free['Peaked'], Kmax), plots_dir, 'Fig4')

    coat = np.load(os.path.join(data_dir, 'phase_barrier.npz'))
    delta, Db_ratio = float(coat['delta']), float(coat['Db_ratio'])   # from the sweep itself
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_Kmax_heatmap(coat['alphas'], coat['u0s'], coat['Kmax'], ax,
                      reference=dict(alphas=alphas, u0s=u0s, Kmax=Kmax), delta=delta, Db_ratio=Db_ratio)
    fig.tight_layout()
    save_figure(fig, plots_dir, 'Fig5')

    print(f"\nK_max contours at u0 = {u0s[0]:.0e}: " + ", ".join(
        f"{lvl} at alpha = {10 ** np.interp(lvl, Kmax[:, 0], np.log10(alphas)):.2f}"
        for lvl in (0.1, 0.3, 0.5, 0.7, 0.9)))
    print(f"Coat: delta_b/R = {delta}, D_b/D = {Db_ratio}, beta = {delta / Db_ratio:g}, "
          f"predicted shift 1 + 2 beta = {1 + 2 * delta / Db_ratio:g}")
    print(f"Done: figures in {plots_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', nargs='?', help='results/<date> folder (default: the newest)')
    folder = ap.parse_args().folder
    if folder is None:
        folder = newest_results_folder()
    elif not os.path.isdir(folder):
        sys.exit(f"No such folder: {folder}")
    main(folder)
