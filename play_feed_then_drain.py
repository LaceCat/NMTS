from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle

from env.physics.thickener import ThickenerModel


@dataclass
class FrameRecord:
    minute: int
    phase: str
    qf: float
    cf: float
    q_uf: float
    concentrations: np.ndarray
    cumulative_feed_t: float
    internal_mass_t: float
    cumulative_discharge_t: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成图表版播放：先正常进料 1 小时，再停进料并只开底流泵直到 10 小时。"
    )
    parser.add_argument("--feed_qf", type=float, default=42.5, help="前半段进料流量 Qf (m^3/h)")
    parser.add_argument("--feed_cf", type=float, default=0.375, help="前半段进料体积分数 Cf")
    parser.add_argument("--feed_minutes", type=int, default=60, help="仅进料阶段时长（分钟）")
    parser.add_argument("--drain_q_uf", type=float, default=50.0, help="排料阶段底流泵流量 Q_uf (m^3/h)")
    parser.add_argument("--total_minutes", type=int, default=600, help="总时长（分钟）")
    parser.add_argument("--frame_stride", type=int, default=2, help="每隔多少分钟保留一帧")
    parser.add_argument("--fps", type=int, default=10, help="GIF 帧率")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "plots" / "feed_then_drain"),
        help="GIF 和 CSV 输出目录",
    )
    return parser.parse_args()


def density_to_concentration_profile(model: ThickenerModel, state: np.ndarray) -> np.ndarray:
    return np.array([max(0.0, model.d2c(value / 1e6)) for value in state], dtype=np.float64)


def internal_mass_t(model: ThickenerModel, state: np.ndarray) -> float:
    layer_volume = model.area * model.detaz
    total = 0.0
    for density_gpm3 in state:
        rho_mix = float(density_gpm3 / 1e6)
        c = float(max(0.0, model.d2c(rho_mix)))
        total += c * rho_mix * layer_volume
    return float(total)


def run_scenario(args: argparse.Namespace) -> tuple[list[FrameRecord], list[list[float]]]:
    model = ThickenerModel()
    state = np.ones(model.N_LAYERS, dtype=np.float64) * 1e6

    frame_stride = max(1, int(args.frame_stride))
    total_minutes = int(args.total_minutes)
    feed_minutes = int(args.feed_minutes)

    history: list[FrameRecord] = []
    csv_rows: list[list[float]] = []
    cumulative_feed_t = 0.0
    cumulative_discharge_t = 0.0

    for minute in range(total_minutes + 1):
        if minute <= feed_minutes:
            phase = "Feed"
            qf = float(args.feed_qf)
            cf = float(args.feed_cf)
            q_uf = 0.0
        else:
            phase = "Drain"
            qf = 0.0
            cf = 0.0
            q_uf = float(args.drain_q_uf)

        concentrations = density_to_concentration_profile(model, state)
        current_internal_mass_t = internal_mass_t(model, state)

        csv_rows.append(
            [
                minute,
                phase,
                qf,
                cf,
                q_uf,
                cumulative_feed_t,
                current_internal_mass_t,
                cumulative_discharge_t,
                *concentrations.tolist(),
            ]
        )

        if minute % frame_stride == 0:
            history.append(
                FrameRecord(
                    minute=minute,
                    phase=phase,
                    qf=qf,
                    cf=cf,
                    q_uf=q_uf,
                    concentrations=concentrations.copy(),
                    cumulative_feed_t=float(cumulative_feed_t),
                    internal_mass_t=float(current_internal_mass_t),
                    cumulative_discharge_t=float(cumulative_discharge_t),
                )
            )

        if minute == total_minutes:
            break

        c_profile, state = model.step(q_uf, qf, cf, state)
        bottom_c = float(c_profile[-1])
        cumulative_feed_t += float(cf * qf * model.c2d(cf) / 60.0) if qf > 0 and cf > 0 else 0.0
        cumulative_discharge_t += float(bottom_c * q_uf * model.c2d(bottom_c) / 60.0) if q_uf > 0 else 0.0

    return history, csv_rows


def build_animation(history: list[FrameRecord], output_path: Path, fps: int) -> None:
    n_layers = 10
    tank_height = 5.79
    tank_width = 2.6
    layer_height = tank_height / n_layers

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.2], height_ratios=[1.0, 1.0])
    ax_tank = fig.add_subplot(gs[:, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    ax_trend = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "浓密机图表版播放：先进料 1 小时，再排料到 10 小时",
        fontsize=14,
        fontweight="bold",
    )

    max_conc = max(float(np.max(record.concentrations)) for record in history)
    max_metric = max(
        max(record.cumulative_feed_t, record.internal_mass_t, record.cumulative_discharge_t)
        for record in history
    )

    cmap = colors.LinearSegmentedColormap.from_list(
        "thickener_layers",
        ["#f2f7fb", "#9bc8e2", "#4e7ea7", "#9a7444", "#5a3017"],
    )

    ax_tank.set_title("浓密机 10 层体积分数")
    ax_tank.set_xlim(-2.2, 2.2)
    ax_tank.set_ylim(-0.6, tank_height + 1.15)
    ax_tank.set_aspect("equal")
    ax_tank.set_xticks([])
    ax_tank.set_ylabel("高度 (m)")
    ax_tank.plot([-tank_width / 2, -tank_width / 2], [0, tank_height], color="black", linewidth=2)
    ax_tank.plot([tank_width / 2, tank_width / 2], [0, tank_height], color="black", linewidth=2)
    ax_tank.plot([-tank_width / 2, tank_width / 2], [0, 0], color="black", linewidth=2)
    ax_tank.text(0.0, tank_height + 0.85, "", ha="center", va="bottom", fontsize=10, color="navy", fontweight="bold")
    info_text = ax_tank.text(0.0, tank_height + 0.48, "", ha="center", va="bottom", fontsize=9, color="#17324d")

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
        rect_texts.append(ax_tank.text(0.0, y + layer_height / 2, "", ha="center", va="center", fontsize=7))

    ax_profile.set_title("10 层体积分数剖面")
    ax_profile.set_xlabel("体积分数")
    ax_profile.set_ylabel("层号")
    ax_profile.set_xlim(0.0, max(0.8, max_conc + 0.05))
    ax_profile.set_ylim(-0.5, 9.5)
    ax_profile.set_yticks(range(10))
    ax_profile.set_yticklabels([f"L{i + 1}" for i in range(10)])
    ax_profile.grid(alpha=0.25)
    profile_line, = ax_profile.plot([], [], "o-", color="#1f77b4", linewidth=2)

    ax_trend.set_title("干矿量随时间变化")
    ax_trend.set_xlabel("时间 (min)")
    ax_trend.set_ylabel("干矿量 (t)")
    ax_trend.set_xlim(0, max(record.minute for record in history))
    ax_trend.set_ylim(0.0, max_metric * 1.08 if max_metric > 0 else 1.0)
    ax_trend.grid(alpha=0.25)
    feed_line, = ax_trend.plot([], [], color="#2a9d8f", linewidth=2.2, label="累计进料")
    internal_line, = ax_trend.plot([], [], color="#e76f51", linewidth=2.2, label="内部干矿量")
    discharge_line, = ax_trend.plot([], [], color="#264653", linewidth=2.2, label="累计排出")
    current_point, = ax_trend.plot([], [], "o", color="#7f0000", markersize=5)
    ax_trend.legend(loc="upper left")

    minutes = [record.minute for record in history]
    feeds = [record.cumulative_feed_t for record in history]
    internals = [record.internal_mass_t for record in history]
    discharges = [record.cumulative_discharge_t for record in history]

    def animate(frame_index: int):
        record = history[frame_index]
        concentrations = record.concentrations

        for idx, value in enumerate(concentrations):
            normalized = np.clip(value / max(max_conc, 1e-12), 0.0, 1.0)
            rects[idx].set_facecolor(cmap(normalized))
            rect_texts[idx].set_text(f"{value:.3f}" if value > 0.005 else "")

        phase_name = "进料阶段" if record.phase == "Feed" else "排料阶段"
        phase_line = (
            f"第 {record.minute} 分钟 | {phase_name} | "
            f"Qf={record.qf:.1f} | Cf={record.cf:.3f} | Q_uf={record.q_uf:.1f}"
        )
        metrics_line = (
            f"累计进料={record.cumulative_feed_t:.2f} t | "
            f"内部干矿量={record.internal_mass_t:.2f} t | "
            f"累计排出={record.cumulative_discharge_t:.2f} t"
        )
        info_text.set_text(f"{phase_line}\n{metrics_line}")

        profile_line.set_data(concentrations, np.arange(n_layers))
        feed_line.set_data(minutes[: frame_index + 1], feeds[: frame_index + 1])
        internal_line.set_data(minutes[: frame_index + 1], internals[: frame_index + 1])
        discharge_line.set_data(minutes[: frame_index + 1], discharges[: frame_index + 1])
        current_point.set_data([record.minute], [record.internal_mass_t])

        return rects + rect_texts + [
            info_text,
            profile_line,
            feed_line,
            internal_line,
            discharge_line,
            current_point,
        ]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(history),
        interval=1000 / max(1, fps),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer="pillow", fps=max(1, fps))
    plt.close(fig)


def save_csv(csv_rows: list[list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "minute",
        "phase",
        "qf",
        "cf",
        "q_uf",
        "cumulative_feed_t",
        "internal_mass_t",
        "cumulative_discharge_t",
    ] + [f"layer_{idx + 1}" for idx in range(10)]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history, csv_rows = run_scenario(args)

    stem = (
        f"feed1h_then_drain10h_"
        f"qf{args.feed_qf:g}_cf{args.feed_cf:g}_"
        f"drain{args.drain_q_uf:g}_total{args.total_minutes}min"
        .replace(".", "p")
    )
    gif_path = output_dir / f"{stem}.gif"
    csv_path = output_dir / f"{stem}.csv"

    build_animation(history, gif_path, fps=args.fps)
    save_csv(csv_rows, csv_path)

    print(f"GIF saved to: {gif_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
