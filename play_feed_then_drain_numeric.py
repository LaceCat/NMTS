from __future__ import annotations

import argparse
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

from play_feed_then_drain import run_scenario, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成中文数字版播放：先进料 1 小时，再只开底流泵直到 10 小时。"
    )
    parser.add_argument("--feed_qf", type=float, default=42.5, help="前 1 小时进料流量 Qf (m^3/h)")
    parser.add_argument("--feed_cf", type=float, default=0.375, help="前 1 小时进料体积分数 Cf")
    parser.add_argument("--feed_minutes", type=int, default=60, help="仅进料阶段时长（分钟）")
    parser.add_argument("--drain_q_uf", type=float, default=50.0, help="排料阶段底流泵流量 Q_uf (m^3/h)")
    parser.add_argument("--total_minutes", type=int, default=600, help="总时长（分钟）")
    parser.add_argument("--frame_stride", type=int, default=2, help="每隔多少分钟保留一帧")
    parser.add_argument("--fps", type=int, default=10, help="GIF 帧率")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "plots" / "feed_then_drain_numeric"),
        help="输出目录",
    )
    return parser.parse_args()


def build_numeric_animation(history, output_path: Path, fps: int) -> None:
    fig = plt.figure(figsize=(10, 7), facecolor="white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    title_text = fig.text(
        0.5,
        0.95,
        "浓密机过程数字播放",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#17324d",
    )

    phase_text = fig.text(
        0.08,
        0.84,
        "",
        ha="left",
        va="top",
        fontsize=14,
        color="#17324d",
        linespacing=1.45,
    )
    metrics_text = fig.text(
        0.08,
        0.62,
        "",
        ha="left",
        va="top",
        fontsize=13,
        color="#2c3e50",
        linespacing=1.5,
    )
    layers_left_text = fig.text(
        0.08,
        0.34,
        "",
        ha="left",
        va="top",
        fontsize=12,
        color="#1f1f1f",
        linespacing=1.45,
    )
    layers_right_text = fig.text(
        0.54,
        0.34,
        "",
        ha="left",
        va="top",
        fontsize=12,
        color="#1f1f1f",
        linespacing=1.45,
    )
    footer_text = fig.text(
        0.5,
        0.05,
        "说明：前 60 分钟只进料，之后停止进料并全开底流泵。",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#5c6770",
    )

    def animate(frame_index: int):
        record = history[frame_index]
        phase_name = "进料阶段" if record.phase == "Feed" else "排料阶段"

        phase_text.set_text(
            "\n".join(
                [
                    f"当前时间：第 {record.minute} 分钟",
                    f"当前阶段：{phase_name}",
                    f"当前操作：Qf = {record.qf:.2f} m^3/h，Cf = {record.cf:.3f}，Q_uf = {record.q_uf:.2f} m^3/h",
                ]
            )
        )

        metrics_text.set_text(
            "\n".join(
                [
                    f"累计进料干矿量：{record.cumulative_feed_t:.2f} t",
                    f"浓密机内部干矿量：{record.internal_mass_t:.2f} t",
                    f"累计排出干矿量：{record.cumulative_discharge_t:.2f} t",
                ]
            )
        )

        left_lines = []
        right_lines = []
        for idx, value in enumerate(record.concentrations, start=1):
            line = f"第{idx}层体积分数：{value:.4f}"
            if idx <= 5:
                left_lines.append(line)
            else:
                right_lines.append(line)
        layers_left_text.set_text("\n".join(left_lines))
        layers_right_text.set_text("\n".join(right_lines))

        return [
            title_text,
            phase_text,
            metrics_text,
            layers_left_text,
            layers_right_text,
            footer_text,
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history, csv_rows = run_scenario(args)
    stem = (
        f"数字版_先进料后排料_"
        f"qf{args.feed_qf:g}_cf{args.feed_cf:g}_"
        f"drain{args.drain_q_uf:g}_total{args.total_minutes}min"
        .replace(".", "p")
    )
    gif_path = output_dir / f"{stem}.gif"
    csv_path = output_dir / f"{stem}.csv"

    build_numeric_animation(history, gif_path, fps=args.fps)
    save_csv(csv_rows, csv_path)

    print(f"GIF saved to: {gif_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
