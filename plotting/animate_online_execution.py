"""Animate online HJB value fields, feedback directions, and UAV trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import infinite_horizon as ih
from _drone_soc_internal import eigen_from_covariance
from drone_soc import compute_collision_gradients, find_pollutant_source
from experiments.exp_pairwise_showcase import _build_case, _sample_scenario
from experiments.exp_paper_discount_sweep import DEFAULT_VARIANT, _runtime


@dataclass
class CycleValue:
    t_start: float
    t_end: float
    value: np.ndarray
    axes: tuple[np.ndarray, ...]
    belief_mu: np.ndarray
    belief_sigma: np.ndarray
    q_log_var: float | None


def _axis_index(axis: np.ndarray, value: float) -> float:
    axis = np.asarray(axis, dtype=float)
    if len(axis) == 1:
        return 0.0
    return float(np.interp(value, axis, np.arange(len(axis), dtype=float)))


def _belief_indices(cycle: CycleValue, first_belief_axis: int) -> list[float]:
    values_log, _ = eigen_from_covariance(cycle.belief_sigma)
    values = [float(values_log[0]), float(values_log[1])]
    if cycle.q_log_var is not None:
        values.append(float(np.log(max(cycle.q_log_var, 1e-12))))
    return [
        _axis_index(cycle.axes[first_belief_axis + offset], value)
        for offset, value in enumerate(values)
    ]


def _slice_value(
    value: np.ndarray,
    axes: tuple[np.ndarray, ...],
    fixed_indices: dict[int, float],
) -> np.ndarray:
    nx, ny = len(axes[0]), len(axes[1])
    coords = np.empty((len(axes), nx, ny), dtype=float)
    coords[0] = np.arange(nx, dtype=float)[:, None]
    coords[1] = np.arange(ny, dtype=float)[None, :]
    for axis_index in range(2, len(axes)):
        coords[axis_index] = fixed_indices[axis_index]
    return map_coordinates(
        value,
        coords.reshape(len(axes), -1),
        order=1,
        mode="nearest",
    ).reshape(nx, ny)


def _one_body_field(cycle: CycleValue, value: np.ndarray) -> np.ndarray:
    belief = _belief_indices(cycle, first_belief_axis=2)
    fixed = {axis: belief[axis - 2] for axis in range(2, len(cycle.axes))}
    return _slice_value(value, cycle.axes, fixed)


def _pairwise_field(
    cycle: CycleValue,
    value: np.ndarray,
    drone_index: int,
    positions: np.ndarray,
) -> np.ndarray:
    belief = _belief_indices(cycle, first_belief_axis=4)
    fields = []
    for other_index in range(len(positions)):
        if other_index == drone_index:
            continue
        other_relative = positions[other_index] - cycle.belief_mu
        fixed = {
            2: _axis_index(cycle.axes[2], float(other_relative[0])),
            3: _axis_index(cycle.axes[3], float(other_relative[1])),
        }
        fixed.update({axis: belief[axis - 4] for axis in range(4, len(cycle.axes))})
        fields.append(_slice_value(value, cycle.axes, fixed))
    return np.mean(fields, axis=0)


def _feedback_field(
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    t: float,
    drone_index: int,
    positions: np.ndarray,
    par: Any,
    problem: Any,
) -> tuple[np.ndarray, np.ndarray]:
    edge_order = 2 if min(field.shape) >= 3 else 1
    grad_x = np.gradient(field, x_axis, axis=0, edge_order=edge_order)
    grad_y = np.gradient(field, y_axis, axis=1, edge_order=edge_order)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    patrol = np.asarray(problem.patrol(np.stack((xx, yy), axis=0), t), dtype=float)
    patrol_norm = np.sqrt(patrol[0] ** 2 + patrol[1] ** 2) + 1e-12
    patrol_scale = np.minimum(1.0, par.v_max / patrol_norm)
    c_eff = float(problem.c1 + problem.c2)
    ux = (
        problem.c1 * problem.wind[0]
        + problem.c2 * patrol[0] * patrol_scale
        - grad_x
    ) / c_eff
    uy = (
        problem.c1 * problem.wind[1]
        + problem.c2 * patrol[1] * patrol_scale
        - grad_y
    ) / c_eff

    if problem.collision is not None and problem.c3 > 0.0:
        collision_gradient = compute_collision_gradients(positions, problem.collision)[drone_index]
        ux -= problem.tau_col * problem.c3 * collision_gradient[0] / c_eff
        uy -= problem.tau_col * problem.c3 * collision_gradient[1] / c_eff

    norm = np.sqrt(ux**2 + uy**2) + 1e-12
    scale = np.minimum(1.0, par.v_max / norm)
    return ux * scale, uy * scale


def _refine(
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor == 1:
        return x_axis, y_axis, field
    dense_x = np.linspace(x_axis[0], x_axis[-1], (len(x_axis) - 1) * factor + 1)
    dense_y = np.linspace(y_axis[0], y_axis[-1], (len(y_axis) - 1) * factor + 1)
    spline = RectBivariateSpline(
        x_axis,
        y_axis,
        field,
        kx=min(3, len(x_axis) - 1),
        ky=min(3, len(y_axis) - 1),
        s=0,
    )
    refined = spline(dense_x, dense_y)
    return dense_x, dense_y, np.clip(refined, np.nanmin(field), np.nanmax(field))


def _positions_at(traj: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(
        [
            [np.interp(t, times, traj[i, coordinate]) for coordinate in range(2)]
            for i in range(traj.shape[0])
        ],
        dtype=float,
    )


def _draw_geometry(ax: plt.Axes, problem: Any) -> None:
    x0, x1 = map(float, problem.domain[0])
    y0, y1 = map(float, problem.domain[1])
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="black", lw=1.2))
    for zone in problem.no_fly_zones or []:
        ax.add_patch(
            Circle(
                tuple(np.asarray(zone.center, dtype=float)),
                float(zone.radius),
                facecolor="0.35",
                edgecolor="black",
                alpha=0.35,
                zorder=5,
            )
        )


def _recorded_run(seed: int, discount: float, method: str, out_dir: Path):
    scenario = _sample_scenario(seed, DEFAULT_VARIANT)
    par, problem, state, fp_solver = _build_case(
        scenario=scenario,
        discount=discount,
        out_dir=str(out_dir.resolve()),
        variant=DEFAULT_VARIANT,
        runtime=_runtime(),
    )
    if method == "order1":
        planner = ih.InfiniteHorizonSingleDronePlanner(wind_aligned=True, fp_solver=fp_solver)
    else:
        planner = ih.InfiniteHorizonTwoDronesPlanner(wind_aligned=True, fp_solver=fp_solver)

    records: list[CycleValue] = []
    original_solve = ih._solve_infinite_hjb_nd_direction

    def recording_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        cycle_state = kwargs["s"]
        t_start, t_end = kwargs["ts"]
        records.append(
            CycleValue(
                t_start=float(t_start),
                t_end=float(t_end),
                value=np.asarray(result[1], dtype=float),
                axes=tuple(np.asarray(axis, dtype=float) for axis in result[2]),
                belief_mu=np.asarray(cycle_state.belief.mu, dtype=float).copy(),
                belief_sigma=np.asarray(cycle_state.belief.sigma, dtype=float).copy(),
                q_log_var=(
                    None
                    if cycle_state.belief.q_log_var is None
                    else float(cycle_state.belief.q_log_var)
                ),
            )
        )
        return result

    ih._solve_infinite_hjb_nd_direction = recording_solve
    try:
        traj, log = find_pollutant_source(par, problem, state, planner=planner)
    finally:
        ih._solve_infinite_hjb_nd_direction = original_solve
    return traj, np.asarray(log["times"], dtype=float), log, records, par, problem


def animate_execution(
    *,
    seed: int,
    discount: float,
    method: str,
    save_path: Path,
    fps: int,
    frames_per_cycle: int,
    contour_refinement: int,
    dpi: int,
) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    traj, times, log, cycles, par, problem = _recorded_run(
        seed, discount, method, save_path.parent
    )
    num_drones = traj.shape[0]
    num_panels = 1 if method == "order1" else num_drones
    colors = [plt.get_cmap("tab10")(i) for i in range(num_drones)]

    frame_specs: list[tuple[int, float, int]] = []
    for cycle_index, cycle in enumerate(cycles):
        for substep in range(frames_per_cycle):
            fraction = substep / frames_per_cycle
            t = cycle.t_start + fraction * (cycle.t_end - cycle.t_start)
            value_index = int(round(fraction * (len(cycle.value) - 1)))
            frame_specs.append((cycle_index, t, value_index))
    final_cycle = cycles[-1]
    frame_specs.append((len(cycles) - 1, final_cycle.t_end, len(final_cycle.value) - 1))

    frames: list[dict[str, Any]] = []
    for cycle_index, t, value_index in frame_specs:
        cycle = cycles[cycle_index]
        positions = _positions_at(traj, times, t)
        x_axis = cycle.axes[0] + cycle.belief_mu[0]
        y_axis = cycle.axes[1] + cycle.belief_mu[1]
        fields = []
        directions = []
        refined = []
        for panel_index in range(num_panels):
            if method == "order1":
                field = _one_body_field(cycle, cycle.value[value_index])
                direction_index = 0
            else:
                field = _pairwise_field(
                    cycle, cycle.value[value_index], panel_index, positions
                )
                direction_index = panel_index
            fields.append(field)
            directions.append(
                _feedback_field(
                    field,
                    x_axis,
                    y_axis,
                    t,
                    direction_index,
                    positions,
                    par,
                    problem,
                )
            )
            refined.append(_refine(field, x_axis, y_axis, contour_refinement))
        frames.append(
            {
                "cycle": cycle_index,
                "time": t,
                "positions": positions,
                "belief_mu": cycle.belief_mu,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "fields": fields,
                "directions": directions,
                "refined": refined,
            }
        )
    num_cycles = len(cycles)
    del cycles

    vmin = min(float(np.nanmin(field)) for frame in frames for field in frame["fields"])
    vmax = max(float(np.nanmax(field)) for frame in frames for field in frame["fields"])
    levels = np.linspace(vmin, vmax if vmax > vmin else vmin + 1.0, 30)
    figsize = (8.0, 7.0) if num_panels == 1 else (18.0, 6.0)
    fig, axes_obj = plt.subplots(1, num_panels, figsize=figsize, squeeze=False)
    axes = list(axes_obj[0])
    first_x, first_y, first_value = frames[0]["refined"][0]
    xx0, yy0 = np.meshgrid(first_x, first_y, indexing="ij")
    initial = axes[0].contourf(xx0, yy0, first_value, levels=levels, cmap="viridis")
    colorbar = fig.colorbar(initial, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Value function")
    title_artist = fig.suptitle("", fontsize=12)

    xlim = tuple(map(float, problem.domain[0]))
    ylim = tuple(map(float, problem.domain[1]))
    source = np.asarray(problem.pollutant.source, dtype=float)
    frame_times = np.asarray([frame["time"] for frame in frames])
    frame_positions = np.asarray([frame["positions"] for frame in frames])

    def update(frame_index: int):
        frame = frames[frame_index]
        for panel_index, ax in enumerate(axes):
            ax.clear()
            dense_x, dense_y, dense_value = frame["refined"][panel_index]
            dense_xx, dense_yy = np.meshgrid(dense_x, dense_y, indexing="ij")
            contour = ax.contourf(
                dense_xx,
                dense_yy,
                dense_value,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                alpha=0.82,
                zorder=1,
            )
            ux, uy = frame["directions"][panel_index]
            magnitude = np.sqrt(ux**2 + uy**2) + 1e-12
            arrow_length = 0.42 * min(
                abs(frame["x_axis"][1] - frame["x_axis"][0]),
                abs(frame["y_axis"][1] - frame["y_axis"][0]),
            )
            xx, yy = np.meshgrid(frame["x_axis"], frame["y_axis"], indexing="ij")
            ax.quiver(
                xx,
                yy,
                ux / magnitude * arrow_length,
                uy / magnitude * arrow_length,
                color="black",
                alpha=0.58,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0025,
                zorder=4,
            )
            _draw_geometry(ax, problem)
            for drone_index in range(num_drones):
                history = frame_positions[: frame_index + 1, drone_index]
                own_panel = method == "order1" or drone_index == panel_index
                ax.plot(
                    history[:, 0],
                    history[:, 1],
                    color=colors[drone_index] if own_panel else "white",
                    alpha=0.9 if own_panel else 0.42,
                    lw=2.0 if own_panel else 1.0,
                    zorder=7,
                )
                position = frame["positions"][drone_index]
                ax.scatter(
                    position[0],
                    position[1],
                    color=colors[drone_index],
                    edgecolors="white",
                    linewidths=1.0,
                    s=78 if own_panel else 42,
                    zorder=9,
                    label=f"UAV {drone_index + 1}" if panel_index == 0 else None,
                )
            ax.scatter(*source, marker="*", s=160, color="red", edgecolors="darkred", zorder=10)
            ax.scatter(
                *frame["belief_mu"], marker="+", s=120, color="darkorange", lw=2.0, zorder=10
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            if method == "order1":
                ax.set_title("Shared one-body value field")
            else:
                ax.set_title(f"Effective pairwise field for UAV {panel_index + 1}")
            ax.grid(True, alpha=0.2)
        axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
        title_artist.set_text(
            f"{method}, seed {seed}, r={discount:g}: cycle {frame['cycle'] + 1}/{num_cycles}, "
            f"t={frame['time']:.1f} s\nArrows show the feedback direction from the value gradient and running-cost corrections",
        )
        return []

    with tempfile.TemporaryDirectory(prefix="online_execution_frames_") as frame_dir:
        frame_pattern = Path(frame_dir) / "frame_%05d.png"
        for frame_index in range(len(frames)):
            update(frame_index)
            fig.savefig(frame_pattern.as_posix() % frame_index, dpi=dpi)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_pattern),
                "-c:v",
                "libx264",
                "-x264-params",
                "keyint=1",
                "-pix_fmt",
                "yuv420p",
                str(save_path),
            ],
            check=True,
        )
    plt.close(fig)
    print(f"Saved {save_path} ({len(frames)} frames; stop={log.get('stop_reason')})")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--discount", type=float, default=0.2)
    parser.add_argument("--method", choices=("order1", "order2"), default="order2")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-cycle", type=int, default=50)
    parser.add_argument("--contour-refinement", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    method_dir = "order1" if args.method == "order1" else "order2"
    default_dir = (
        Path("results/paper_pairwise_discount_sweep")
        / f"discount_{args.discount:.3f}".replace(".", "p")
        / f"seed_{args.seed:03d}"
        / method_dir
        / "execution_animation"
    )
    output = args.output or default_dir / f"seed{args.seed:03d}_{method_dir}_r{args.discount:g}_execution.mp4"
    animate_execution(
        seed=args.seed,
        discount=args.discount,
        method=args.method,
        save_path=output,
        fps=args.fps,
        frames_per_cycle=args.frames_per_cycle,
        contour_refinement=args.contour_refinement,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
