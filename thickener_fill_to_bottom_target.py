"""
Simulate thickener filling from an all-zero 10-layer concentration profile
until the bottom-layer concentration reaches a target value.

Default assumption:
- typical feed: Qf = 42.5 m^3/h, Cf = 0.375
- no underflow withdrawal during the fill-up diagnostic: Q_uf = 0.0 m^3/h

Outputs:
- a GIF animation of the filling process
- a CSV file with minute-by-minute layer concentrations
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle

from env.physics.thickener import ThickenerModel


warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")


@dataclass
class FrameRecord:
    minute: int
    concentrations: np.ndarray
    bottom_concentration: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill the thickener from zero profile until bottom concentration reaches a target."
    )
    parser.add_argument("--qf", type=float, default=42.5, help="Feed flow rate Qf in m^3/h.")
    parser.add_argument("--cf", type=float, default=0.375, help="Feed volume fraction Cf.")
    parser.add_argument("--q_uf", type=float, default=0.0, help="Underflow rate Q_uf in m^3/h.")
    parser.add_argument("--target", type=float, default=0.66, help="Target bottom concentration.")
    parser.add_argument(
        "--max_minutes",
        type=int,
        default=5000,
        help="Maximum simulated minutes before aborting.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=5,
        help="Keep one animation frame every N simulated minutes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "plots"),
        help="Directory for generated outputs.",
    )
    return parser.parse_args()


def density_to_concentration_profile(model: ThickenerModel, state: np.ndarray) -> np.ndarray:
    concentrations = []
    for value in state:
        if value <= 1e-12:
            concentrations.append(0.0)
        else:
            concentrations.append(max(0.0, model.d2c(value / 1e6)))
    return np.array(concentrations, dtype=np.float64)


def run_simulation(args) -> tuple[list[FrameRecord], list[list[float]], float | None]:
    model = ThickenerModel()
    state = np.zeros(model.N_LAYERS, dtype=np.float64)

    frame_stride = max(1, int(args.frame_stride))
    history: list[FrameRecord] = []
    csv_rows: list[list[float]] = []
    crossing_time: float | None = None

    prev_bottom = 0.0

    for minute in range(args.max_minutes + 1):
        concentrations = density_to_concentration_profile(model, state)
        bottom = float(concentrations[-1])

        csv_rows.append([minute, bottom, *concentrations.tolist()])
        if minute % frame_stride == 0:
            history.append(FrameRecord(minute, concentrations.copy(), bottom))

        if bottom >= args.target:
            if minute == 0 or bottom <= prev_bottom:
                crossing_time = float(minute)
            else:
                ratio = (args.target - prev_bottom) / max(bottom - prev_bottom, 1e-12)
                crossing_time = (minute - 1) + float(np.clip(ratio, 0.0, 1.0))

            if history[-1].minute != minute:
                history.append(FrameRecord(minute, concentrations.copy(), bottom))
            break

        prev_bottom = bottom
        _, state = model.step(args.q_uf, args.qf, args.cf, state)

    return history, csv_rows, crossing_time


def build_animation(
    history: list[FrameRecord],
    target: float,
    qf: float,
    cf: float,
    q_uf: float,
    output_path: Path,
):
    n_layers = 10
    tank_height = 5.79
    tank_width = 2.6
    layer_height = tank_height / n_layers

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35])
    ax_tank = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    ax_trend = fig.add_subplot(gs[0, 2])

    fig.suptitle(
        f"Thickener Fill-Up From Zero Profile | Qf={qf:.2f} m^3/h, Cf={cf:.3f}, Q_uf={q_uf:.2f}",
        fontsize=13,
        fontweight="bold",
    )

    cmap = colors.LinearSegmentedColormap.from_list(
        "slurry_fill",
        ["#eef6fb", "#9fd2e9", "#4f8fba", "#b2763f", "#6f3d1d"],
    )

    ax_tank.set_title("10-Layer Tank View")
    ax_tank.set_xlim(-1.9, 1.9)
    ax_tank.set_ylim(-0.4, tank_height + 0.9)
    ax_tank.set_aspect("equal")
    ax_tank.set_xticks([])
    ax_tank.set_ylabel("Height (m)")
    ax_tank.plot([-tank_width / 2, -tank_width / 2], [0, tank_height], color="black", linewidth=2)
    ax_tank.plot([tank_width / 2, tank_width / 2], [0, tank_height], color="black", linewidth=2)
    ax_tank.plot([-tank_width / 2, tank_width / 2], [0, 0], color="black", linewidth=2)
    ax_tank.plot([-0.25, 0.25], [4.0, 4.0], color="green", linewidth=3)
    ax_tank.text(0.0, 4.15, "Feed", ha="center", va="bottom", fontsize=9, color="green")
    ax_tank.text(0.0, -0.18, "Bottom layer", ha="center", va="top", fontsize=9, color="maroon")

    rects = []
    rect_texts = []
    for idx in range(n_layers):
        y = idx * layer_height
        rect = Rectangle(
            (-tank_width / 2, y),
            tank_width,
            layer_height,
            facecolor=cmap(0.0),
            edgecolor="gray",
            linewidth=0.7,
        )
        ax_tank.add_patch(rect)
        rects.append(rect)
        text = ax_tank.text(0.0, y + layer_height / 2, "", ha="center", va="center", fontsize=7)
        rect_texts.append(text)

    info_text = ax_tank.text(
        0.0,
        tank_height + 0.48,
        "",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="navy",
    )

    ax_profile.set_title("Layer Concentration Profile")
    ax_profile.set_xlabel("Concentration")
    ax_profile.set_ylabel("Layer index")
    ax_profile.set_xlim(0.0, max(0.72, target + 0.05))
    ax_profile.set_ylim(-0.5, 9.5)
    ax_profile.set_yticks(range(10))
    ax_profile.set_yticklabels([f"L{i + 1}" for i in range(10)])
    ax_profile.axvline(target, color="red", linestyle="--", alpha=0.7, label=f"Target = {target:.2f}")
    ax_profile.grid(alpha=0.25)
    ax_profile.legend(loc="lower right")
    profile_line, = ax_profile.plot([], [], "o-", color="#1f77b4", linewidth=2)

    ax_trend.set_title("Bottom Concentration vs Time")
    ax_trend.set_xlabel("Time (min)")
    ax_trend.set_ylabel("Bottom concentration")
    ax_trend.set_xlim(0, max(record.minute for record in history))
    ax_trend.set_ylim(0.0, max(target + 0.05, max(record.bottom_concentration for record in history) + 0.03))
    ax_trend.axhline(target, color="red", linestyle="--", alpha=0.7)
    ax_trend.grid(alpha=0.25)
    trend_line, = ax_trend.plot([], [], color="#d2691e", linewidth=2.2)
    current_point, = ax_trend.plot([], [], "o", color="#8b0000", markersize=6)

    minutes = [record.minute for record in history]
    bottoms = [record.bottom_concentration for record in history]

    def animate(frame_index):
        record = history[frame_index]
        concentrations = record.concentrations

        for idx, value in enumerate(concentrations):
            normalized = np.clip(value / max(target, 1e-12), 0.0, 1.0)
            rects[idx].set_facecolor(cmap(normalized))
            rect_texts[idx].set_text(f"{value:.3f}" if value > 0.005 else "")

        info_text.set_text(
            f"Minute {record.minute} | Bottom C = {record.bottom_concentration:.4f} | Target = {target:.2f}"
        )

        profile_line.set_data(concentrations, np.arange(n_layers))
        trend_line.set_data(minutes[: frame_index + 1], bottoms[: frame_index + 1])
        current_point.set_data([record.minute], [record.bottom_concentration])

        return rects + rect_texts + [info_text, profile_line, trend_line, current_point]

    fps = 10
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(history),
        interval=1000 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


def save_csv(csv_rows: list[list[float]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["minute", "bottom_concentration"] + [f"layer_{idx + 1}" for idx in range(10)]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(csv_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history, csv_rows, crossing_time = run_simulation(args)

    stem = (
        f"fill_zero_to_bottom_{str(args.target).replace('.', 'p')}"
        f"_qf{str(args.qf).replace('.', 'p')}"
        f"_cf{str(args.cf).replace('.', 'p')}"
        f"_quf{str(args.q_uf).replace('.', 'p')}"
    )
    gif_path = output_dir / f"{stem}.gif"
    csv_path = output_dir / f"{stem}.csv"

    save_csv(csv_rows, csv_path)
    build_animation(
        history=history,
        target=args.target,
        qf=args.qf,
        cf=args.cf,
        q_uf=args.q_uf,
        output_path=gif_path,
    )

    if crossing_time is None:
        print(f"Target bottom concentration {args.target:.3f} was not reached within {args.max_minutes} minutes.")
    else:
        print(f"Target bottom concentration {args.target:.3f} reached at about {crossing_time:.2f} minutes.")
    print(f"GIF saved to: {gif_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
