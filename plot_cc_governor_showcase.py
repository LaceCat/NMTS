from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent / "__agent_debug__" / "cc_governor_compare"
    cc = json.loads((root / "cc_governor_compare.json").read_text(encoding="utf-8"))
    pid = json.loads((root / "pid_cc_current_env_5seed.json").read_text(encoding="utf-8"))

    controllers = [
        ("CC-TD3\nNo Governor", cc["td3_cc_no_governor"]["summary"]),
        ("CC-TD3\nWith Governor", cc["td3_cc_with_governor"]["summary"]),
        ("CC-PID", pid["summary"]),
    ]
    labels = [c[0] for c in controllers]
    energy_mean = [c[1]["energy_mean"] for c in controllers]
    energy_std = [c[1].get("energy_std", 0.0) for c in controllers]
    mass_mean = [c[1]["mass_mean"] for c in controllers]
    mass_std = [c[1].get("mass_std", 0.0) for c in controllers]
    zero_unsafe = [c[1]["zero_unsafe_rate"] for c in controllers]
    inband = [c[1]["inband_rate"] for c in controllers]

    colors = ["#b55d4c", "#3b7ea1", "#4c9a5f"]
    accent = "#d9e6f2"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("CC Control Strategy Comparison", fontsize=18, fontweight="bold")

    x = np.arange(len(labels))

    ax = axes[0, 0]
    ax.bar(x, energy_mean, yerr=energy_std, color=colors, capsize=6, alpha=0.9)
    ax.set_title("(a) Energy Cost", fontweight="bold")
    ax.set_ylabel("Cost-weighted energy")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(energy_mean):
        ax.text(i, v + max(energy_std) + 6, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    ax = axes[0, 1]
    ax.axhspan(400, 420, color=accent, alpha=0.7, label="Target band 400-420 t")
    ax.axhline(400, color="#4d4d4d", linestyle="--", linewidth=1.2)
    ax.bar(x, mass_mean, yerr=mass_std, color=colors, capsize=6, alpha=0.9)
    ax.set_title("(b) Final Dry Mass", fontweight="bold")
    ax.set_ylabel("Final mass (t)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for i, v in enumerate(mass_mean):
        ax.text(i, v + max(mass_std) + 0.8, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    ax = axes[1, 0]
    width = 0.34
    ax.bar(x - width / 2, zero_unsafe, width=width, color="#5c7cfa", label="Zero-unsafe rate")
    ax.bar(x + width / 2, inband, width=width, color="#82c91e", label="In-band rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("(c) Safety and Target-band Hit Rate", fontweight="bold")
    ax.set_ylabel("Rate")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for i, (u, b) in enumerate(zip(zero_unsafe, inband)):
        ax.text(i - width / 2, u + 0.03, f"{u * 100:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, b + 0.03, f"{b * 100:.0f}%", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    ax.axhspan(400, 420, color=accent, alpha=0.7)
    for label, summary, color in zip(labels, [c[1] for c in controllers], colors):
        ax.errorbar(
            summary["energy_mean"],
            summary["mass_mean"],
            xerr=summary.get("energy_std", 0.0),
            yerr=summary.get("mass_std", 0.0),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=5,
            markersize=9,
            label=label.replace("\n", " "),
        )
        ax.annotate(
            label.replace("\n", " "),
            (summary["energy_mean"], summary["mass_mean"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )
    ax.set_title("(d) Energy-Mass Trade-off", fontweight="bold")
    ax.set_xlabel("Cost-weighted energy")
    ax.set_ylabel("Final mass (t)")
    ax.grid(alpha=0.25)

    note = (
        "Tail governor keeps TD3 safe while reducing energy from 703.1 to 672.8 "
        "and pulling mass from 438.9 t to 400.5 t.\n"
        "PID remains the strongest low-energy baseline in the current environment."
    )
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=10)

    out = root / "cc_governor_showcase.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
