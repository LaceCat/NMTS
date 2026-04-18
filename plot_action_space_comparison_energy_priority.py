from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "action_space_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EpochPick:
    mode: str
    run_name: str
    epoch: int
    stage: str
    avg_reward: float
    avg_mass: float
    avg_energy: float
    unsafe_episode_rate: float
    unsafe_step_rate: float
    completion_rate: float
    inband_rate: float
    avg_target_band_distance: float
    selection_note: str


def _iter_mode_epochs(mode: str) -> list[dict]:
    rows: list[dict] = []
    prefix = f"{mode.lower()}_"
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir() or not run_dir.name.lower().startswith(prefix):
            continue
        log_path = run_dir / "training_log.json"
        if not log_path.exists():
            continue
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for ep in data.get("epochs", []):
            rows.append(
                {
                    "run_name": run_dir.name,
                    "epoch": int(ep.get("epoch", 0)),
                    "stage": str(ep.get("stage", "")),
                    "avg_reward": float(ep.get("avg_reward", -1e9)),
                    "avg_mass": float(ep.get("avg_mass", 0.0)),
                    "avg_energy": float(ep.get("avg_energy", 1e9)),
                    "unsafe_episode_rate": float(ep.get("unsafe_episode_rate", 1.0)),
                    "unsafe_step_rate": float(ep.get("unsafe_step_rate", 1.0)),
                    "completion_rate": float(ep.get("completion_rate", 0.0)),
                    "inband_rate": float(ep.get("inband_rate", ep.get("pass_rate", 0.0))),
                    "avg_target_band_distance": float(ep.get("avg_target_band_distance", 1e9)),
                }
            )
    return rows


def _pick_epoch(mode: str) -> EpochPick:
    rows = _iter_mode_epochs(mode)
    if not rows:
        raise RuntimeError(f"No epoch history found for mode={mode}")

    # Energy-priority selection:
    # 1. absolutely safe
    # 2. at least reach 400 t on average (completion_rate == 1.0)
    # 3. then minimize energy
    # 4. target-band closeness is descriptive only, used as a late tie-break
    safe_rows = [r for r in rows if r["unsafe_episode_rate"] == 0.0]
    feasible_rows = [r for r in safe_rows if r["completion_rate"] >= 1.0]

    if not feasible_rows:
        candidate_rows = safe_rows
        note = "safe-only fallback"
    else:
        candidate_rows = feasible_rows
        note = "safe + completion + min energy"

    best = min(
        candidate_rows,
        key=lambda r: (
            float(r["avg_energy"]),
            float(r["avg_target_band_distance"]),
            -float(r["avg_reward"]),
        ),
    )

    return EpochPick(
        mode=mode,
        run_name=best["run_name"],
        epoch=best["epoch"],
        stage=best["stage"],
        avg_reward=best["avg_reward"],
        avg_mass=best["avg_mass"],
        avg_energy=best["avg_energy"],
        unsafe_episode_rate=best["unsafe_episode_rate"],
        unsafe_step_rate=best["unsafe_step_rate"],
        completion_rate=best["completion_rate"],
        inband_rate=best["inband_rate"],
        avg_target_band_distance=best["avg_target_band_distance"],
        selection_note=note,
    )


def _save_csv(path: Path, picks: list[EpochPick]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(picks[0]).keys()))
        writer.writeheader()
        for item in picks:
            writer.writerow(asdict(item))


def _save_json(path: Path, picks: list[EpochPick]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(item) for item in picks], f, indent=2, ensure_ascii=False)


def _save_figure(path: Path, picks: list[EpochPick]) -> None:
    modes = [p.mode for p in picks]
    mass = [p.avg_mass for p in picks]
    energy = [p.avg_energy for p in picks]
    reward = [p.avg_reward for p in picks]
    safe = [p.unsafe_episode_rate for p in picks]
    inband = [p.inband_rate * 100.0 for p in picks]
    completion = [p.completion_rate * 100.0 for p in picks]
    colors = ["#6B8E23", "#D98E04", "#2C7FB8"]
    x = np.arange(len(modes))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_energy = fig.add_subplot(gs[0, 0])
    ax_mass = fig.add_subplot(gs[0, 1])
    ax_rate = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    bars = ax_energy.bar(x, energy, color=colors, width=0.6)
    ax_energy.set_xticks(x, modes)
    ax_energy.set_ylabel("Total Energy Cost")
    ax_energy.set_title("(a) Energy-Priority Comparison")
    for bar, value in zip(bars, energy):
        ax_energy.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    bars = ax_mass.bar(x, mass, color=colors, width=0.6)
    ax_mass.axhspan(400, 420, color="#9FD3C7", alpha=0.25, label="Target Band 400-420 t")
    ax_mass.set_xticks(x, modes)
    ax_mass.set_ylabel("Final Dry Mass (t)")
    ax_mass.set_title("(b) Mass of the Selected Low-Energy Models")
    ax_mass.legend(loc="upper left")
    for bar, value in zip(bars, mass):
        ax_mass.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    width = 0.24
    ax_rate.bar(x - width, [100.0 * (1.0 - s) for s in safe], width=width, color="#4CAF50", label="Safe Rate")
    ax_rate.bar(x, completion, width=width, color="#FFB300", label="Completion Rate")
    ax_rate.bar(x + width, inband, width=width, color="#E64A19", label="In-Band Rate")
    ax_rate.set_xticks(x, modes)
    ax_rate.set_ylim(0, 105)
    ax_rate.set_ylabel("Rate (%)")
    ax_rate.set_title("(c) Safety and Completion of the Selected Models")
    ax_rate.legend(loc="upper right")

    ax_table.axis("off")
    cell_text = []
    for p in picks:
        cell_text.append(
            [
                p.mode,
                p.run_name,
                p.epoch,
                f"{p.avg_mass:.1f}",
                f"{p.avg_energy:.1f}",
                f"{p.inband_rate*100:.0f}%",
            ]
        )
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Mode", "Run", "Epoch", "Mass (t)", "Energy", "In-Band"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.08, 1.7)
    ax_table.set_title("(d) Selected Epochs Under Energy-Priority Criterion")

    fig.suptitle(
        "DD / CD / CC Comparison Rebuilt for the Actual Thesis Objective\n"
        "Safety first, then energy minimization, while keeping the 400 t requirement",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    picks = [_pick_epoch(mode) for mode in ("DD", "CD", "CC")]
    csv_path = OUTPUT_DIR / "action_space_comparison_energy_priority.csv"
    json_path = OUTPUT_DIR / "action_space_comparison_energy_priority.json"
    fig_path = OUTPUT_DIR / "action_space_comparison_energy_priority.png"
    _save_csv(csv_path, picks)
    _save_json(json_path, picks)
    _save_figure(fig_path, picks)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved FIG: {fig_path}")


if __name__ == "__main__":
    main()
