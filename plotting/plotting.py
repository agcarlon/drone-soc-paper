import os
import sys
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from _drone_soc_internal import eigen_from_covariance  # noqa: E402

# ── Paper-wide style ─────────────────────────────────────────────────────────
PAPER_STYLE: dict = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.unicode_minus": False,
}
plt.rcParams.update(PAPER_STYLE)

# Consistent colour palette (Vega-10 first two entries, then standard cycle)
PAPER_COLORS: list[str] = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b"]


def styled_boxplot(
    ax,
    data: list,
    labels: list[str],
    colors: list[str] | None = None,
    log_y: bool = False,
    ylabel: str | None = None,
    title: str | None = None,
) -> dict:
    """Render a paper-style boxplot with filled, coloured boxes.

    Parameters
    ----------
    ax:
        Matplotlib ``Axes`` to draw on.
    data:
        One array-like per group.
    labels:
        Tick labels, one per group.
    colors:
        Face colours for boxes; defaults to ``PAPER_COLORS``.
    log_y:
        Apply log scale to the y-axis.
    ylabel, title:
        Optional axis label and title.

    Returns
    -------
    dict
        The raw ``boxplot`` return dict (``bp``).
    """
    if colors is None:
        colors = PAPER_COLORS[: len(data)]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
    )
    n = len(data)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % n])
        patch.set_alpha(0.6)
    for element in ("medians", "whiskers", "caps", "fliers"):
        items = bp[element]
        for j, item in enumerate(items):
            # whiskers and caps have 2 items per group; medians and fliers have 1
            group_idx = (j // 2) if element in ("whiskers", "caps") else j
            item.set_color(colors[group_idx % n])
    if log_y:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    return bp


def plot_convergence_multiple(logs, par_list, problem_list, path):

    # Collect all belief errors and traces in lists
    all_belief_errors = []
    all_traces = []
    all_times = None
    
    for log, par, problem in zip(logs, par_list, problem_list):
        pol = problem.pollutant
        beliefs = log['beliefs']
        mus = np.array([b.mu for b in beliefs])
        sigmas = np.stack([b.sigma for b in beliefs])
        measurement_times = np.array([0., *par.meas_times])
        all_times = measurement_times  # assuming same time vector across runs
    
        belief_errors = np.linalg.norm(mus - pol.source, axis=1)**2
        traces = np.trace(sigmas, axis1=1, axis2=2)
    
        # Pad to full length if shorter
        if len(belief_errors) < len(measurement_times):
            pad_len = len(measurement_times) - len(belief_errors)
            belief_errors = np.concatenate([belief_errors, np.full(pad_len, belief_errors[-1])])
        if len(traces) < len(measurement_times):
            pad_len = len(measurement_times) - len(traces)
            traces = np.concatenate([traces, np.full(pad_len, traces[-1])])
    
        all_belief_errors.append(belief_errors)
        all_traces.append(traces)
    
    # Convert to arrays (shape: n_runs x n_times)
    all_belief_errors = np.array(all_belief_errors)
    all_traces = np.array(all_traces)
    
    # Compute statistics
    belief_median = np.median(all_belief_errors, axis=0)
    belief_q25 = np.quantile(all_belief_errors, 0.25, axis=0)
    belief_q75 = np.quantile(all_belief_errors, 0.75, axis=0)
    
    trace_median = np.median(all_traces, axis=0)
    trace_q25 = np.quantile(all_traces, 0.25, axis=0)
    trace_q75 = np.quantile(all_traces, 0.75, axis=0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    
    # --- Belief error ---
    for run in all_belief_errors:
        ax1.semilogy(all_times[:len(run)], run, color='blue', alpha=0.15, linewidth=1)
    
    # Fill between 25% and 75%
    ax1.fill_between(all_times, belief_q25, belief_q75, color='blue', alpha=0.2, label=r'Inner 50\%')
    
    # Median line
    ax1.semilogy(all_times, belief_median, 'b-', linewidth=2, label='Median')
    
    ax1.set_ylabel(r'$\| \mu - \theta_t \|^2$')
    ax1.set_title('Belief Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Covariance trace ---
    for run in all_traces:
        ax2.semilogy(all_times[:len(run)], run, color='green', alpha=0.15, linewidth=1)
    
    ax2.fill_between(all_times, trace_q25, trace_q75, color='green', alpha=0.2, label=r'Inner 50\%')
    ax2.semilogy(all_times, trace_median, 'g-', linewidth=2, label='Median')
    ax2.set_xlim([measurement_times[0],measurement_times[-1]])
    ax2.set_xlabel('Measurement Time')
    ax2.set_ylabel(r'tr$(\Sigma)$')
    ax2.set_title('Covariance Trace')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(path, "all_conv.pdf"))
    plt.close(fig)


def _get_output_dir(par, out_dir: Optional[str] = None) -> str:
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    default_out = par.output_dir if getattr(par, "output_dir", None) else "."
    os.makedirs(default_out, exist_ok=True)
    return default_out


def _compute_traj_bounds(traj: np.ndarray, source: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    num_drones = traj.shape[0]
    try:
        pts = traj.reshape(num_drones, 2, -1)
        xs = pts[:, 0, :].ravel()
        ys = pts[:, 1, :].ravel()
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]
        x_min = float(np.min(xs)) if xs.size else float(source[0])
        x_max = float(np.max(xs)) if xs.size else float(source[0])
        y_min = float(np.min(ys)) if ys.size else float(source[1])
        y_max = float(np.max(ys)) if ys.size else float(source[1])
        x_min = min(x_min, float(source[0]))
        x_max = max(x_max, float(source[0]))
        y_min = min(y_min, float(source[1]))
        y_max = max(y_max, float(source[1]))
        dx = max(1e-9, x_max - x_min)
        dy = max(1e-9, y_max - y_min)
        pad = 0.08
        return x_min - pad * dx, x_max + pad * dx, y_min - pad * dy, y_max + pad * dy
    except Exception:
        return None


def _add_pollutant_background(ax, par, problem, x_bounds, y_bounds):
    pol = problem.pollutant
    try:
        x0, x1 = x_bounds
        y0, y1 = y_bounds
        # Keep this a true filled-contour plot for consistency with the
        # standard diagnostics, but avoid oversized PDF contour paths.
        n_grid = 120
        xs = np.linspace(x0, x1, n_grid, dtype=float)
        ys = np.linspace(y0, y1, n_grid, dtype=float)
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
        t_bg = float(par.meas_times[0]) if getattr(par, "meas_times", None) is not None and len(par.meas_times) > 0 else 0.0
        C = pol.get_pollutant_concentration(pts, t_bg, theta=np.asarray(pol.source), intensity=getattr(pol, "q", None))
        C = np.asarray(C, dtype=float).reshape(X.shape)
        eps = 1e-30
        logC = np.log10(np.maximum(C, eps))
        vmin = float(np.floor(np.nanmin(logC)))
        vmax = float(np.ceil(np.nanmax(logC)))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return None, None
        levels = np.arange(vmin, vmax + 1.0, 1.0)
        if len(levels) < 2:
            return None, None
        from matplotlib.colors import BoundaryNorm, ListedColormap
        colors = plt.cm.Greens(np.linspace(0.25, 0.95, len(levels) - 1))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        m = ax.contourf(X, Y, logC, levels=levels, cmap=cmap, norm=norm, antialiased=False, alpha=0.65, zorder=0)
        return m, levels
    except Exception:
        return None, None


def plot_drone_paths(
    traj,
    log,
    par,
    problem,
    s,
    out_dir: Optional[str] = None,
    show_patrol_reference: bool = False,
    show_full_domain: bool = False,
):
    """Plot only drone trajectories (+ optional patrol reference, no-fly, wind, plume background)."""
    pol = problem.pollutant
    out_dir = _get_output_dir(par, out_dir)
    traj = np.asarray(traj, dtype=float)
    num_drones = traj.shape[0]
    if show_full_domain and getattr(problem, "domain", None) is not None:
        domain = np.asarray(problem.domain, dtype=float)
        bounds = (float(domain[0, 0]), float(domain[0, 1]), float(domain[1, 0]), float(domain[1, 1]))
    else:
        bounds = _compute_traj_bounds(traj, np.asarray(pol.source, dtype=float))

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))
    m_bg = None
    levels_bg = None
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        m_bg, levels_bg = _add_pollutant_background(ax1, par, problem, (x_min, x_max), (y_min, y_max))

    ts = np.array(log["times"])
    meas_times = par.meas_times
    meas_idxs = [np.argmin(np.abs(t_ - ts)) for t_ in meas_times]

    for i in range(num_drones):
        drone_traj = traj[i, :, :]
        ax1.plot(drone_traj[0, :], drone_traj[1, :], c=f"C{i}", label=f"Drone {i+1}", linewidth=1.5)
        ax1.scatter(
            drone_traj[0, meas_idxs],
            drone_traj[1, meas_idxs],
            c=f"C{i}",
            marker="o",
            s=5,
            linewidth=3,
            label=f"Drone {i+1} Measurement",
            zorder=5,
        )
        if show_patrol_reference and getattr(s.drones[i], "_patrol_dir_fn", None) is not None:
            drone_patrol = [s.drones[i].patrolling_trajectory(t) for t in ts]
            drone_patrol = np.stack(drone_patrol, axis=-1)
            ax1.plot(drone_patrol[0, :], drone_patrol[1, :], "--", c=f"C{i}", label=f"Drone {i+1} Patrol", linewidth=1.0)

    zones = getattr(problem, "no_fly_zones", None)
    if zones is not None:
        first = True
        for z in zones:
            circle = plt.Circle(z.center, z.radius, color="gray", alpha=0.50, label=("No-fly zone" if first else None), zorder=2)
            ax1.add_patch(circle)
            first = False

    ax1.scatter(pol.source[0], pol.source[1], color="red", marker="x", s=100, linewidth=3, label="Source", zorder=5)
    wind = np.array(problem.wind)
    ax1.quiver(pol.source[0], pol.source[1], wind[0], wind[1], angles="xy", scale_units="xy", scale=0.1, color="C4", label="Wind")

    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)

    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Drone Trajectories")
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    if m_bg is not None and levels_bg is not None:
        try:
            cbar = fig.colorbar(m_bg, ax=ax1, fraction=0.046, pad=0.02, ticks=levels_bg)
            cbar.set_label(r"$\log_{10}$ concentration", rotation=90)
        except Exception:
            pass

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out_dir, "drones_path.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_cooperative_experiment_setting(
    par,
    problem,
    s,
    source_box_x: tuple[float, float],
    source_box_y: tuple[float, float],
    out_path: str,
):
    """
    Plot the static geometry of the cooperative order-comparison experiment.

    The figure shows:
    - domain boundary
    - initial drone positions
    - no-fly zones
    - source spawn box
    - one illustrative source at the box center
    - plume log-concentration contours for that source
    - wind direction
    """
    pol = problem.pollutant
    domain = np.asarray(problem.domain, dtype=float)
    drone_positions = np.asarray(s.drone_positions, dtype=float)
    source_center = np.array(
        [
            0.5 * (float(source_box_x[0]) + float(source_box_x[1])),
            0.5 * (float(source_box_y[0]) + float(source_box_y[1])),
        ],
        dtype=float,
    )

    original_source = np.asarray(pol.source, dtype=float).copy()
    pol.source = source_center.copy()

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.4))
    m_bg, levels_bg = _add_pollutant_background(
        ax,
        par,
        problem,
        (float(domain[0, 0]), float(domain[0, 1])),
        (float(domain[1, 0]), float(domain[1, 1])),
    )

    domain_rect = Rectangle(
        (float(domain[0, 0]), float(domain[1, 0])),
        float(domain[0, 1] - domain[0, 0]),
        float(domain[1, 1] - domain[1, 0]),
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(domain_rect)

    zones = getattr(problem, "no_fly_zones", None) or []
    first_zone = True
    for z in zones:
        circle = plt.Circle(
            z.center,
            z.radius,
            color="gray",
            alpha=0.45,
            label=("No-fly zone" if first_zone else None),
            zorder=4,
        )
        ax.add_patch(circle)
        first_zone = False

    source_box = Rectangle(
        (float(source_box_x[0]), float(source_box_y[0])),
        float(source_box_x[1] - source_box_x[0]),
        float(source_box_y[1] - source_box_y[0]),
        fill=False,
        edgecolor="C1",
        linestyle="--",
        linewidth=2.0,
        label="Source location box",
        zorder=5,
    )
    ax.add_patch(source_box)

    ax.scatter(
        drone_positions[:, 0],
        drone_positions[:, 1],
        c="C0",
        s=65,
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        label="Initial drones",
        zorder=6,
    )
    for idx, pos in enumerate(drone_positions, start=1):
        ax.text(
            float(pos[0]) + 8.0,
            float(pos[1]) + 8.0,
            f"{idx}",
            color="C0",
            fontsize=11,
            weight="bold",
            zorder=7,
        )

    ax.scatter(
        source_center[0],
        source_center[1],
        color="red",
        marker="x",
        s=110,
        linewidth=2.5,
        label="Illustrative source",
        zorder=7,
    )
    wind = np.asarray(problem.wind, dtype=float)
    ax.quiver(
        source_center[0],
        source_center[1],
        wind[0],
        wind[1],
        angles="xy",
        scale_units="xy",
        scale=0.08,
        color="C4",
        width=0.0045,
        label="Wind",
        zorder=7,
    )

    ax.set_xlim(float(domain[0, 0]), float(domain[0, 1]))
    ax.set_ylim(float(domain[1, 0]), float(domain[1, 1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    if m_bg is not None and levels_bg is not None:
        cbar = fig.colorbar(m_bg, ax=ax, fraction=0.046, pad=0.02, ticks=levels_bg)
        cbar.set_label(r"$\log_{10}$ concentration")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    pol.source = original_source


def plot_belief_distribution_ellipses(log, par, problem, out_dir: Optional[str] = None):
    """Plot belief means with 1-sigma ellipses and wind at true source."""
    pol = problem.pollutant
    out_dir = _get_output_dir(par, out_dir)
    beliefs = log["beliefs"]
    mus = np.array([b.mu for b in beliefs])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(mus[:, 0], mus[:, 1], "ko--", linewidth=2, markersize=6, label=r"$\mu$")
    ax.scatter(pol.source[0], pol.source[1], color="red", marker="x", s=100, linewidth=3, label="Source", zorder=5)

    for k, bel in enumerate(beliefs, start=1):
        vals_log, ang = eigen_from_covariance(bel.sigma)
        vals = np.exp(vals_log)
        vals = vals[vals.argsort()[::-1]]
        width, height = 2 * np.sqrt(vals)
        ellipse = Ellipse(
            xy=bel.mu,
            width=width,
            height=height,
            angle=np.degrees(ang),
            edgecolor="C0",
            facecolor="none",
            linestyle="--",
            linewidth=1.5,
            label=r"1-$\sigma$ ellipse" if k == 1 else None,
            zorder=1,
        )
        ax.add_patch(ellipse)

    wind = np.array(problem.wind)
    ax.quiver(pol.source[0], pol.source[1], wind[0], wind[1], angles="xy", scale_units="xy", scale=0.1, color="C4", label="Wind")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Belief Distribution with 1-std Ellipses")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, "problem_dist_with_ellipses.pdf"))
    plt.close(fig)


def plot_confidence_ellipses_with_wind(log, par, problem, out_dir: Optional[str] = None):
    """Plot centered confidence ellipses for each belief and wind direction at origin."""
    out_dir = _get_output_dir(par, out_dir)
    beliefs = log["beliefs"]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Confidence ellipses and the wind")

    scale = 1.0
    for j, bel in enumerate(beliefs):
        vals_log, ang = eigen_from_covariance(bel.sigma)
        vals = np.exp(vals_log)
        vals = vals[vals.argsort()[::-1]]
        sds = np.sqrt(vals)
        width, height = 2 * np.abs(sds)
        ellipse = Ellipse(
            xy=np.zeros(2),
            width=width,
            height=height,
            angle=np.degrees(ang),
            edgecolor="C2",
            facecolor="none",
            linestyle="--",
            linewidth=1,
        )
        if j == 0:
            ellipse.set_label(r"$1\sigma$")
            scale = max(float(np.min(vals)), 1e-12)
        ax1.add_patch(ellipse)

    wind = np.array(problem.wind)
    ax1.quiver(0, 0, wind[0], wind[1], angles="xy", scale_units="xy", scale=0.5 / scale, color="C3", label="Wind")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("x[m]")
    ax1.set_ylabel("y[m]")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_ellipses_with_wind.pdf"))
    plt.close(fig)


def plot_path(traj, log, par, problem, s, **kwargs):
    """
    Backward-compatible alias for trajectory plotting.

    This function now creates only the path figure. Use dedicated functions for
    additional plots:
    - `plot_belief_convergence_with_intensity`
    - `plot_belief_distribution_ellipses`
    - `plot_confidence_ellipses_with_wind`
    - `plot_running_cost_breakdown`
    """
    plot_drone_paths(traj, log, par, problem, s, **kwargs)


def plot_concentration_contours(
    pol,
    times: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    n_grid: int,
    levels: int = 40,
):
    """
    Draw filled-contour maps of the pollutant plume at one or more times.

    Args:
        times (list[float]): Simulation times (seconds) at which to plot C(x, t).
        x_bounds (tuple): (xmin, xmax) plotting domain.
        y_bounds (tuple): (ymin, ymax) plotting domain.
        n_grid (int): Number of grid points per axis.
        pol (Pollutant): Pollutant model.
        levels (int): Number of contour levels.

    Returns:
        list: List of (fig, ax) tuples for further customisation.
    """
    import matplotlib.pyplot as plt

    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)

    figs_axes = []
    for t in times:
        conc = pol.get_pollutant_concentration(coords, t, theta=pol.source).reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(6, 5))
        cf = ax.contourf(X, Y, conc, levels=levels, cmap="viridis")
        ax.set_title(f"Pollutant concentration at t = {t:.1f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(cf, ax=ax, label="C(x,t)")
        figs_axes.append((fig, ax))

    return figs_axes


def plot_belief_convergence_with_intensity(log, par, problem, out_dir: Optional[str] = None):
    """
    Plot belief error, covariance trace, and intensity mean vs. true intensity.
    Expects `log["beliefs"]` entries as `Belief` dataclasses.
    """
    pol = problem.pollutant
    if not log.get("beliefs"):
        return
    beliefs = log["beliefs"]
    mus = np.stack([b.mu for b in beliefs], axis=0)
    sigmas = np.stack([b.sigma for b in beliefs], axis=0)
    q_logs_raw = [b.q_log for b in beliefs]
    q_log_vars_raw = [b.q_log_var for b in beliefs]
    q_logs = np.array([np.nan if v is None else float(v) for v in q_logs_raw], dtype=float)
    q_log_vars = np.array([np.nan if v is None else float(v) for v in q_log_vars_raw], dtype=float)
    if np.all(np.isnan(q_logs)):
        q_logs = None
    if np.all(np.isnan(q_log_vars)):
        q_log_vars = None

    measurement_times = np.array([0., *par.meas_times])

    belief_errors = np.linalg.norm(mus - pol.source, axis=1)
    traces = np.trace(sigmas, axis1=1, axis2=2)
    # Align time axis with available beliefs (prior + one per measurement)
    t_bel = measurement_times[: len(belief_errors)]

    if out_dir is None:
        out_dir = par.output_dir if getattr(par, "output_dir", None) else "."
    os.makedirs(out_dir, exist_ok=True)

    if q_logs is not None:
        fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1, ax2 = axes[0], axes[1]

    ax1.semilogy(t_bel, belief_errors[: len(t_bel)], "bo-", linewidth=2, markersize=5)
    ax1.set_ylabel(r"$\| \mu_\theta - \theta \|$")
    ax1.set_title("Belief error")
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t_bel, traces[: len(t_bel)], "go-", linewidth=2, markersize=5)
    ax2.set_ylabel(r"tr$(\Sigma_{\theta})$")
    ax2.set_title("Covariance trace (position)")
    # Tolerance line (used by early-stopping logic)
    tol = getattr(par, "tol_sigma", None)
    if tol is not None:
        try:
            tol_val = float(tol)
            if tol_val > 0.0:
                ax2.axhline(tol_val, color="k", linestyle="--", linewidth=1.5, alpha=0.9)
        except Exception:
            pass
    ax2.grid(True, alpha=0.3)

    if q_logs is not None:
        ax3 = axes[2]
        q_mean = np.exp(q_logs)
        # ax3.plot(measurement_times, q_mean, "ro-", label="E[q]")
        # ax3.axhline(pol.q, color="k", linestyle="--", label="true q")
        # if q_log_vars is not None:
        #     q_std = np.exp(q_logs + 0.5 * q_log_vars) - np.exp(q_logs - 0.5 * q_log_vars)
        #     ax3.fill_between(measurement_times, np.maximum(q_mean - q_std, 0.0), q_mean + q_std, color="r", alpha=0.2, label="±std (approx)")
        t_q = t_bel[: len(q_mean)]
        ax3.semilogy(t_q, np.abs(q_mean[: len(t_q)] - pol.q), "ro-", linewidth=2, markersize=5)
        ax3.set_ylabel(r"$|\mu_q - q|$")
        ax3.set_xlabel("Time (s)")
        ax3.set_title("Intensity belief")
        ax3.grid(True, alpha=0.3)
        # ax3.legend()
    else:
        ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "belief_convergence_intensity.pdf"))
    plt.close(fig)


def default_after_measurement_callback(
    y: np.ndarray,
    positions: np.ndarray,
    t: float,
    par,
    problem,
    s,
    prior,
    posterior,
    used_all_measurements: bool,
) -> None:
    """
    Default callback for `find_pollutant_source(..., on_after_measurement=...)`.
    """
    intensity = None
    if posterior.q_log is not None:
        intensity = float(np.exp(posterior.q_log))
    plot_prior_lkl_post_contours(
        y=y,
        x=positions,
        t=t,
        par=par,
        problem=problem,
        prior=(prior.mu, prior.sigma),
        max_post=posterior.mu,
        intensity=intensity,
    )


def plot_running_cost_breakdown(
    traj: np.ndarray,
    log: dict,
    par,
    problem,
    *,
    drone_idx: Optional[int] = None,
    out_dir: Optional[str] = None,
    make_path_plot: bool = False,
):
    """
    Plot a per-drone breakdown of the *realized running cost* along the executed trajectory.

    This uses the logged controls and positions (realized trajectory), and mirrors the
    running cost structure used in the HJB setup:
      - effort/fuel term (c1)
      - patrol deviation term (c2)
      - collision potential term (c3) attributed to the chosen drone
      - no-fly zone potential term (c4) for the chosen drone

    Notes:
    - Collision is a *pairwise potential*. For a single drone i we attribute:
        phi_coll_i = 0.5 * sum_{j!=i} f(||x_i - x_j||)
      so that sum_i phi_coll_i = sum_{i<j} f(d_ij).
    - No-fly is per-drone: phi_nf_i = sum_z f_z(x_i).
    - Patrol/collision are read from `problem.patrol` and `problem.collision`.
    """
    if out_dir is None:
        out_dir = par.output_dir if getattr(par, "output_dir", None) else "."
    os.makedirs(out_dir, exist_ok=True)

    if ("controls" not in log) or (len(log["controls"]) == 0):
        print("plot_running_cost_breakdown: no logged controls found in log['controls']; skipping.")
        return

    times = np.asarray(log.get("times", []), dtype=float)
    controls = np.asarray(log["controls"], dtype=float)  # (T, N, 2)
    traj = np.asarray(traj, dtype=float)  # (N, 2, Ttraj)

    T = min(controls.shape[0], traj.shape[2], len(times))
    if T <= 1:
        print("plot_running_cost_breakdown: not enough samples; skipping.")
        return

    N = traj.shape[0]
    # Select which drones to plot
    if drone_idx is None:
        drone_indices = list(range(N))
    else:
        drone_indices = [int(np.clip(int(drone_idx), 0, N - 1))]

    pos = traj[:, :, :T]          # (N,2,T)
    u = controls[:T, :, :]        # (T,N,2)
    t = times[:T]                 # (T,)

    patrol_fn = getattr(problem, "patrol", None)
    if patrol_fn is None:
        raise RuntimeError("plot_running_cost_breakdown requires ProblemSpec.patrol.")

    wind = np.asarray(problem.wind, dtype=float)
    v_max = float(getattr(par, "v_max", 1.0))

    # Per-time-step components for each requested drone: shape (M, T)
    M = len(drone_indices)
    eff = np.zeros((M, T), dtype=float)
    pat = np.zeros((M, T), dtype=float)
    coll = np.zeros((M, T), dtype=float)
    nf = np.zeros((M, T), dtype=float)

    c1 = float(problem.c1)
    c2 = float(problem.c2)
    c3 = float(problem.c3)
    c4 = float(problem.c4)

    zones = getattr(problem, "no_fly_zones", None) or []
    collision = getattr(problem, "collision", None)

    for k in range(T):
        for m, idx in enumerate(drone_indices):
            ui = u[k, idx]
            xi = pos[idx, :, k]
            # effort/fuel
            du_w = ui - wind
            eff[m, k] = 0.5 * c1 * float(np.dot(du_w, du_w))

            # patrol deviation (clip u_p to v_max like the HJB setup)
            u_p = np.asarray(patrol_fn(np.asarray(xi, dtype=float), float(t[k])), dtype=float).reshape(2)
            upn = float(np.linalg.norm(u_p)) + 1e-12
            if upn > v_max:
                u_p = u_p * (v_max / upn)
            du_p = ui - u_p
            pat[m, k] = 0.5 * c2 * float(np.dot(du_p, du_p))

            # collision potential attributed to drone idx
            if (collision is not None) and (c3 > 0.0) and (N >= 2):
                others = np.delete(pos[:, :, k], idx, axis=0)  # (N-1,2)
                dists = np.linalg.norm(others - xi[None, :], axis=1)
                coll[m, k] = 0.5 * c3 * float(np.sum(collision.function(dists)))

            # no-fly potential for drone idx
            if (zones is not None) and (len(zones) > 0) and (c4 > 0.0):
                nf_val = 0.0
                for z in zones:
                    try:
                        nf_val += float(z.function(np.asarray(xi, dtype=float)))
                    except Exception:
                        continue
                nf[m, k] = c4 * nf_val

    total = eff + pat + coll + nf

    # ---- Figure: percentage breakdown for all requested drones (stacked to 100%) ----
    import matplotlib.pyplot as plt

    eps = 1e-18
    denom = np.maximum(total, eps)
    frac_eff = eff / denom
    frac_pat = pat / denom
    frac_coll = coll / denom
    frac_nf = nf / denom
    mask_zero = total <= eps
    if np.any(mask_zero):
        frac_eff = frac_eff.copy(); frac_eff[mask_zero] = 0.0
        frac_pat = frac_pat.copy(); frac_pat[mask_zero] = 0.0
        frac_coll = frac_coll.copy(); frac_coll[mask_zero] = 0.0
        frac_nf = frac_nf.copy(); frac_nf[mask_zero] = 0.0

    labels = ["effort (c1)", "patrol (c2)", "collision (c3)", "no-fly (c4)"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig_h = max(2.0 * M, 4.0)
    fig, axes = plt.subplots(M, 1, figsize=(10, fig_h), sharex=True)
    if M == 1:
        axes = [axes]

    for m, idx in enumerate(drone_indices):
        ax = axes[m]
        ax.stackplot(
            t,
            frac_eff[m],
            frac_pat[m],
            frac_coll[m],
            frac_nf[m],
            colors=colors,
            alpha=0.88,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"Drone {idx}")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.0)

    # Shared x-axis formatting: force exact [0, T] without padding
    # T_plot = float(getattr(par, "t_end", t[-1]))
    T_plot = t[-1]
    for ax in axes:
        ax.set_xlim(0.0, T_plot)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Running cost breakdown (percent of total per time step)")

    # Single legend for the whole figure
    handles = [plt.Line2D([0], [0], color=c, linewidth=8) for c in colors]
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9, framealpha=0.9)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out_dir, "running_cost_breakdown_all_drones.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Path plot intentionally removed (user preferred to focus on the breakdown plot).
    return


def plot_prior_lkl_post_contours(y, x, t, par, problem, prior, max_post, intensity=None):
    """
    Three-panel contour: negative log prior, negative log likelihood, and negative log posterior (unnormalised).
    Saves to par.output_dir/contours.
    """
    pol = problem.pollutant
    # grids over theta
    x_grid, y_grid = np.linspace(*problem.domain[0], 100), np.linspace(*problem.domain[1], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    thetas = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # negative log prior (up to additive constant)
    diff = thetas - prior[0]
    prior_quad = np.einsum('ni,ij,nj->n', diff, np.linalg.inv(prior[1]), diff)
    neg_log_prior = 0.5 * prior_quad.reshape(100, 100)

    # negative log likelihood (sum over observations, up to additive constant)
    neg_log_lkl = np.zeros_like(neg_log_prior)
    for x_i, y_i in zip(x, y):
        # If intensity belief is enabled, passing `intensity` (e.g., MAP q) makes this
        # likelihood consistent with the posterior used by the optimizer.
        x_eval = np.repeat(np.asarray(x_i, dtype=float).reshape(1, 2), thetas.shape[0], axis=0)
        preds = pol.get_pollutant_concentration(x_eval, t, thetas, intensity=intensity)  # vectorised in theta
        resid = (y_i - preds)
        neg_log_lkl += (0.5 / (par.noise_std ** 2)) * (resid.reshape(100, 100) ** 2)

    # negative log posterior (unnormalised)
    neg_log_post = neg_log_prior + neg_log_lkl
    # Stabilize exp(-neg_log_*) to avoid underflow (common on realistic scales).
    # Shifting by min keeps argmax unchanged and makes each surface have max=1.
    def _stable_exp_minus(neg_log: np.ndarray) -> np.ndarray:
        m = float(np.nanmin(neg_log))
        z = -(neg_log - m)
        z = np.clip(z, -700.0, 0.0)  # float64 exp underflow guard
        return np.exp(z)

    post = _stable_exp_minus(neg_log_post)
    prior = _stable_exp_minus(neg_log_prior)
    lkl = _stable_exp_minus(neg_log_lkl)

    # Plot with separate scaling + colorbar per panel (prevents likelihood/posterior from looking flat)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    c0 = axes[0].contourf(X, Y, prior, levels=30, cmap="viridis")
    axes[0].set_title('prior')
    axes[0].scatter(x[:, 0], x[:, 1], c='k', marker='x')
    fig.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04)

    c1 = axes[1].contourf(X, Y, lkl, levels=30, cmap="viridis")
    axes[1].set_title('likelihood')
    axes[1].scatter(x[:, 0], x[:, 1], c='k', marker='x')
    axes[1].scatter(*pol.source, c='g', marker='x')
    fig.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04)

    c2 = axes[2].contourf(X, Y, post, levels=30, cmap="viridis")
    axes[2].set_title('posterior (unnormalised)')
    axes[2].scatter(x[:, 0], x[:, 1], c='k', marker='x', label="Measurements")
    axes[2].scatter(*max_post, c='r', marker='x', label="MAP")
    axes[2].scatter(*pol.source, c='g', marker='x', label="Source")
    axes[2].legend()
    fig.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel('X')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('Y')


    fig.suptitle(f"Distributions at time {t:.2f}s")
    fig.tight_layout()

    base = par.output_dir if getattr(par, 'output_dir', None) else "."
    contours_dir = os.path.join(base, "contours")
    os.makedirs(contours_dir, exist_ok=True)
    n = len(y)
    # fig.savefig(os.path.join(contours_dir, f"triple_posterior_{t:.2f}s.pdf"))
    fig.savefig(os.path.join(contours_dir, f"triple_posterior_{t:.2f}s.png"), dpi=300)
    plt.close(fig)


def plot_concentration_contours_with_drones(
    problem,
    drone_pos,
    times: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    n_grid: int,
    noise_std: float = None
):
    """
    Draw filled-contour maps of the pollutant plume in log scale at one or more times. With drones 
    and the signal to noise ratio at the drone position

    Args:
        drone_pos np.array: shape(no_drones, 2, no_measurements) of drone positions
        times (list[float]): Simulation times (seconds) at which to plot C(x, t).
        x_bounds (tuple): (xmin, xmax) plotting domain.
        y_bounds (tuple): (ymin, ymax) plotting domain.
        n_grid (int): Number of grid points per axis.
        problem (ProblemSpec): Problem specification.
        levels (int): Number of contour levels.

    Returns:
        list: List of (fig, ax) tuples for further customisation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    
    
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X, Y], axis=-1).reshape(-1, 2)
    

    

    drone_pos = np.atleast_3d(drone_pos)
    
    
    figs_axes = []
    
    
    for i, time in enumerate(times):
        
        if noise_std is not None:
            conc_at_drone = pol.get_pollutant_concentration(drone_pos[:,:,i], 0., theta=pol.source)
            snr = 10 * np.log10(np.mean(np.square(conc_at_drone))/np.square(noise_std))
        
        conc = pol.get_pollutant_concentration(coords, time, theta=pol.source).reshape(n_grid, n_grid)
        conc_plot = conc.copy()
        conc_plot[conc_plot < 1e-6] = 1e-6
        fig, ax = plt.subplots(figsize=(6, 5))
        levels = 10 ** np.arange(np.floor(np.log10(conc_plot.min())), np.ceil(np.log10(conc_plot.max()+1e-5)) + 1)
        cf = ax.contourf(X, Y, conc_plot, levels=levels, alpha=0.8, cmap='Greens',norm=mcolors.LogNorm(vmin=conc_plot.min(), vmax=conc_plot.max()))
        ax.scatter(*drone_pos[:,:,i].T, color='black', marker='o', s=15, label='Drones' if noise_std is None else r'Drones $SNR_{dB}: %.1f$'%snr)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        
        cbar = fig.colorbar(cf, ax=ax, label="C(x,t)")
        cbar.set_ticks(levels)
        cbar.ax.set_yticklabels([f"1e{int(np.log10(l))}" for l in levels])
        ax.text(
            0.98, 0.02,              
            f"t: {time:.2f}",       
            transform=ax.transAxes,  
            ha='right', va='bottom', 
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                alpha=0.7
            )
        )
        figs_axes.append((fig, ax))

    return figs_axes

def animate_value_function(
    V_time: np.ndarray,
    grid_axes: Tuple[np.ndarray, ...],
    spatial_axes: Tuple[int, int] = (0, 1),
    fixed_state_indices: Optional[dict] = None,
    mark_drone_indices: Optional[tuple] = None,
    times: Optional[np.ndarray] = None,
    levels: int = 30,
    cmap: str = "viridis",
    interval: Optional[int] = None,
    repeat: bool = False,
    save_path: Optional[str] = None,
    video_length: float = 10.0,
    show_quiver: bool = True,
    quiver_stride: Optional[int] = None,
    quiver_scale: Optional[float] = None,
    quiver_color: str = "black",
    quiver_alpha: float = 0.8,
    quiver_width: float = 0.0025,
    quiver_negate: bool = False,
    quiver_normalize: bool = True,
    quiver_length: Optional[float] = None,
    quiver_clip_percentile: float = 95.0,
):
    """
    Animate the evolution of the value function V over time on a 2D spatial slice.

    Parameters
    ----------
    V_time : np.ndarray
        Array of shape (Nt+1, *shape) returned inside the HJB solver, holding V at
        each time step over the state grid defined by `grid_axes`.
    grid_axes : tuple of np.ndarray
        The tensor grid axes for the state dimensions (no time axis).
    spatial_axes : tuple of int
        Which two state axes to visualize (default: (0, 1)). These should be spatial
        position axes corresponding to X and Y on the plane.
    fixed_state_indices : dict, optional
        Mapping from state-axis index -> integer grid index to use when slicing all
        non-visualized axes (e.g., belief parameters). Any axis not specified will be
        fixed at its middle index.
    mark_drone_indices : tuple, optional
        Marker for an "other drone" (or any reference point) on the 2D slice.
        Supported forms:
        - (i, j) where i and j are integer indices on the plotted spatial axes.
        - (x, y) where x and y are coordinates in the plotted spatial units.
    times : np.ndarray, optional
        Optional array of times (length Nt+1). If not provided, frame indices are used.
    levels : int
        Number of contour levels.
    cmap : str
        Matplotlib colormap name.
    interval : int
        Delay between frames in milliseconds. If None, it is chosen so that
        the total animation duration is approximately ``video_length`` seconds.
    repeat : bool
        Whether the animation repeats after the last frame.
    save_path : str, optional
        If provided, path where to save the animation (e.g., 'value_evolution.mp4' or
        'value_evolution.gif'). The writer is inferred by Matplotlib.
    video_length : float
        Desired total duration of the animation in seconds (used when
        ``interval`` is None).
    show_quiver : bool
        If True, overlay a quiver plot of the spatial gradient of V on the 2D slice.
    quiver_stride : int
        Subsampling stride for quiver arrows (larger = fewer arrows). If None,
        a stride is chosen so that there are ~10 arrows per spatial dimension.
    quiver_scale : float, optional
        Passed to Matplotlib `ax.quiver(scale=...)`. If None, Matplotlib chooses.
    quiver_color : str
        Arrow color for the gradient quiver.
    quiver_alpha : float
        Alpha for quiver arrows.
    quiver_width : float
        Arrow width for quiver arrows.
    quiver_negate : bool
        If True, plot -∇V instead of ∇V (sometimes convenient since controls often
        align with negative gradients).
    quiver_normalize : bool
        If True (default), normalize the gradient field so arrows are short and
        comparable in length (direction field). This also makes temporal evolution
        of the field easier to see.
    quiver_length : float, optional
        Target arrow length in data units when `quiver_normalize=True`. If None,
        it is set to ~0.3 of the quiver grid spacing.
    quiver_clip_percentile : float
        Percentile for clipping gradient magnitudes before normalization (avoids
        a few extreme values dominating the scaling).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The created animation object.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib import animation

    assert V_time.ndim == 1 + len(grid_axes), "V_time and grid_axes dimensionality mismatch"

    # Build indexers for a 2D slice on the chosen spatial axes; others fixed
    num_state_axes = len(grid_axes)
    ax_x, ax_y = spatial_axes
    assert 0 <= ax_x < num_state_axes and 0 <= ax_y < num_state_axes and ax_x != ax_y

    if fixed_state_indices is None:
        fixed_state_indices = {}

    # Prepare static meshgrid for the two spatial axes
    X_axis = grid_axes[ax_x]
    Y_axis = grid_axes[ax_y]
    X, Y = np.meshgrid(X_axis, Y_axis, indexing="ij")
    # Grid spacing for gradient computation on the slice
    dx = float(X_axis[1] - X_axis[0]) if len(X_axis) > 1 else 1.0
    dy = float(Y_axis[1] - Y_axis[0]) if len(Y_axis) > 1 else 1.0
    Lx = float(X_axis[-1] - X_axis[0]) if len(X_axis) > 1 else 1.0
    Ly = float(Y_axis[-1] - Y_axis[0]) if len(Y_axis) > 1 else 1.0

    def _quiver_stride_value() -> int:
        """
        Choose a stride so quiver density is ~10 arrows per spatial dimension.
        If user provides quiver_stride, respect it.
        """
        if quiver_stride is not None:
            return max(1, int(quiver_stride))
        nx = max(1, len(X_axis))
        ny = max(1, len(Y_axis))
        # ~10 arrows per axis => stride ~ N/10
        sx = max(1, int(round(nx / 20)))
        sy = max(1, int(round(ny / 20)))
        # Use a single stride (same in both directions) for simplicity
        return int(max(1, min(sx, sy)))

    def _prepare_quiver_field(Vk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (U, V) arrays for quiver arrows on the full slice grid.
        """
        dVdx, dVdy = np.gradient(Vk, dx, dy, edge_order=1)
        if quiver_negate:
            dVdx = -dVdx
            dVdy = -dVdy
        if not quiver_normalize:
            return dVdx, dVdy

        mag = np.sqrt(dVdx * dVdx + dVdy * dVdy)
        # Clip extreme magnitudes before normalization for better visuals
        if np.any(np.isfinite(mag)) and float(quiver_clip_percentile) > 0:
            clip_val = float(np.nanpercentile(mag, float(quiver_clip_percentile)))
            if np.isfinite(clip_val) and clip_val > 0:
                mag = np.minimum(mag, clip_val)
        mag_safe = mag + 1e-12
        ux = dVdx / mag_safe
        uy = dVdy / mag_safe

        s = _quiver_stride_value()
        spacing = max(abs(dx) * s, abs(dy) * s, 1e-12)
        target_len = float(quiver_length) if (quiver_length is not None) else float(0.3 * spacing)
        # Scale unit vectors to a short, consistent length in data units
        return ux * target_len, uy * target_len

    # Precompute global vmin/vmax across time for consistent color scale
    Nt_plus_1 = V_time.shape[0]

    def _slice_at_time(t_idx: int) -> np.ndarray:
        indexers: list = []
        for i in range(num_state_axes):
            if i == ax_x or i == ax_y:
                indexers.append(slice(None))
            else:
                if i in fixed_state_indices:
                    idx = int(fixed_state_indices[i])
                else:
                    idx = int(len(grid_axes[i]) // 2)
                indexers.append(idx)
        Vt = V_time[t_idx]
        V_slice = Vt[tuple(indexers)]
        # Ensure orientation matches X (rows) x Y (cols)
        return V_slice

    vmin = np.inf
    vmax = -np.inf
    for k in range(Nt_plus_1):
        V_slice = _slice_at_time(k)
        vmin = min(vmin, float(np.nanmin(V_slice)))
        vmax = max(vmax, float(np.nanmax(V_slice)))

    # Guard against degenerate or non-finite ranges
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if not (vmax > vmin):
        eps = 1e-6 if vmax == 0.0 else 1e-6 * abs(vmax)
        vmin -= eps
        vmax += eps

    # Use fixed contour levels over the entire animation for stable colors
    if isinstance(levels, int):
        levels_array = np.linspace(vmin, vmax, max(2, levels))
    else:
        levels_array = levels

    # If interval not provided, choose it to match desired video_length
    if interval is None:
        interval = int(max(1, round(1000.0 * video_length / max(1, Nt_plus_1))))

    fig, ax = plt.subplots(figsize=(6, 5))

    def _scatter_mark() -> None:
        """
        Scatter the mark point on the current axis, if requested.
        """
        if mark_drone_indices is None:
            return
        if len(mark_drone_indices) != 2:
            return
        a0, a1 = mark_drone_indices[0], mark_drone_indices[1]
        # (i, j) index form
        if isinstance(a0, (int, np.integer)) and isinstance(a1, (int, np.integer)):
            i = int(a0)
            j = int(a1)
            if 0 <= i < X.shape[0] and 0 <= j < X.shape[1]:
                ax.scatter(X[i, j], Y[i, j], color="red", marker="x")
            return
        # (x, y) coordinate form
        try:
            x0 = float(a0)
            y0 = float(a1)
            ax.scatter(x0, y0, color="red", marker="x")
        except Exception:
            return

    def _format_title(k: int) -> str:
        if times is not None and len(times) == Nt_plus_1:
            return f"V at t={times[k]:.2f}"
        return f"V at frame {k}"

    # Initial frame
    V0 = _slice_at_time(0)
    cf = ax.contourf(X, Y, V0, levels=levels_array, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax)
    qv = None
    if show_quiver:
        # Compute gradient field for this frame (depends on V(t))
        dVdx, dVdy = _prepare_quiver_field(V0)
        stride = _quiver_stride_value()
        Xq = X[::stride, ::stride]
        Yq = Y[::stride, ::stride]
        Uq = dVdx[::stride, ::stride]
        Vq = dVdy[::stride, ::stride]
        qv = ax.quiver(
            Xq,
            Yq,
            Uq,
            Vq,
            color=quiver_color,
            alpha=float(quiver_alpha),
            width=float(quiver_width),
            scale=quiver_scale,
        )
    _scatter_mark()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(_format_title(0))

    def update(k: int):
        nonlocal cf, qv
        # Remove previous contour collections robustly
        try:
            collections = list(getattr(cf, "collections", []))
            for coll in collections:
                coll.remove()
        except Exception:
            # Fallback: clear all artist collections on the axis (except colorbar axis)
            for coll in list(ax.collections):
                try:
                    coll.remove()
                except Exception:
                    pass
        # Remove previous quiver (if any)
        if qv is not None:
            try:
                qv.remove()
            except Exception:
                pass
            qv = None
        Vk = _slice_at_time(k)
        cf = ax.contourf(X, Y, Vk, levels=levels_array, vmin=vmin, vmax=vmax, cmap=cmap)
        if show_quiver:
            # Recompute gradient field for this frame so quiver evolves with V(t)
            dVdx, dVdy = _prepare_quiver_field(Vk)
            stride = _quiver_stride_value()
            Xq = X[::stride, ::stride]
            Yq = Y[::stride, ::stride]
            Uq = dVdx[::stride, ::stride]
            Vq = dVdy[::stride, ::stride]
            qv = ax.quiver(
                Xq,
                Yq,
                Uq,
                Vq,
                color=quiver_color,
                alpha=float(quiver_alpha),
                width=float(quiver_width),
                scale=quiver_scale,
            )
        _scatter_mark()
        # Keep colorbar in sync with new mappable
        try:
            cbar.update_normal(cf)
        except Exception:
            pass
        ax.set_title(_format_title(k))
        return getattr(cf, "collections", [])

    anim = animation.FuncAnimation(
        fig, update, frames=range(Nt_plus_1), interval=interval, blit=False, repeat=repeat
    )

    if save_path is not None:
        try:
            ext = os.path.splitext(save_path)[1].lower()
            if ext in (".gif", ".gifv"):
                try:
                    writer = animation.PillowWriter(fps=max(1, int(1000 / max(1, interval))))
                    anim.save(save_path, writer=writer, dpi=150)
                except Exception as e:
                    print(f"Warning: failed to save GIF to {save_path}: {e}")
            elif ext in (".mp4", ".m4v", ".mov"):
                try:
                    writer = animation.FFMpegWriter(fps=max(1, int(1000 / max(1, interval))))
                    anim.save(save_path, writer=writer, dpi=150)
                except Exception as e:
                    # Fallback to GIF
                    alt_path = os.path.splitext(save_path)[0] + ".gif"
                    print("MovieWriter ffmpeg unavailable; using Pillow instead.")
                    try:
                        writer = animation.PillowWriter(fps=max(1, int(1000 / max(1, interval))))
                        anim.save(alt_path, writer=writer, dpi=150)
                        print(f"Saved fallback animation to {alt_path}")
                    except Exception as e2:
                        print(f"Warning: failed to save animation to {save_path} and fallback GIF {alt_path}: {e2}")
            else:
                # Unknown extension: try Pillow GIF as safe default
                alt_path = os.path.splitext(save_path)[0] + ".gif"
                try:
                    writer = animation.PillowWriter(fps=max(1, int(1000 / max(1, interval))))
                    anim.save(alt_path, writer=writer, dpi=150)
                    print(f"Saved GIF animation to {alt_path}")
                except Exception as e:
                    print(f"Warning: failed to save animation to {alt_path}: {e}")
        except Exception as e:
            print(f"Warning: failed to save animation: {e}")
    plt.close(fig)
    return anim
    pol = problem.pollutant
    pol = problem.pollutant
