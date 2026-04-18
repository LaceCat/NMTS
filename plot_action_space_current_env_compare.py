from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEBUG_ROOT = ROOT / "__agent_debug__"


def load_data():
    current_env = json.loads((DEBUG_ROOT / "action_space_current_env_compare.json").read_text(encoding="utf-8"))
    cc_governor = json.loads((DEBUG_ROOT / "cc_governor_compare" / "cc_governor_compare.json").read_text(encoding="utf-8"))

    dd = current_env["dd_current_env"]["summary"]
    cd = current_env["cd_current_env"]["summary"]
    cc = cc_governor["td3_cc_with_governor"]["summary"]

    return {
        "DD": {
            "mass_mean": dd["mass_mean"],
            "mass_std": dd["mass_std"],
            "energy_mean": dd["energy_mean"],
            "energy_std": dd["energy_std"],
            "zero_unsafe_rate": dd["zero_unsafe_rate"],
            "completion_rate": dd["completion_rate"],
            "inband_rate": dd["inband_rate"],
        },
        "CD": {
            "mass_mean": cd["mass_mean"],
            "mass_std": cd["mass_std"],
            "energy_mean": cd["energy_mean"],
            "energy_std": cd["energy_std"],
            "zero_unsafe_rate": cd["zero_unsafe_rate"],
            "completion_rate": cd["completion_rate"],
            "inband_rate": cd["inband_rate"],
        },
        "CC": {
            "mass_mean": cc["mass_mean"],
            "mass_std": cc["mass_std"],
            "energy_mean": cc["energy_mean"],
            "energy_std": cc["energy_std"],
            "zero_unsafe_rate": cc["zero_unsafe_rate"],
            "completion_rate": cc["completion_rate"],
            "inband_rate": cc["inband_rate"],
        },
    }


def main():
    summary = load_data()
    labels = ["DD", "CD", "CC"]
    colors = ["#6c757d", "#f4a261", "#2a9d8f"]
    x = np.arange(len(labels))

    mass = np.array([summary[k]["mass_mean"] for k in labels], dtype=float)
    mass_std = np.array([summary[k]["mass_std"] for k in labels], dtype=float)
    energy = np.array([summary[k]["energy_mean"] for k in labels], dtype=float)
    energy_std = np.array([summary[k]["energy_std"] for k in labels], dtype=float)
    zero_unsafe = np.array([summary[k]["zero_unsafe_rate"] for k in labels], dtype=float)
    completion = np.array([summary[k]["completion_rate"] for k in labels], dtype=float)
    inband = np.array([summary[k]["inband_rate"] for k in labels], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DD / CD / CC Comparison Under Current Environment", fontsize=16, fontweight="bold")

    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.bar(x, energy, yerr=energy_std, color=colors, alpha=0.92, capsize=6)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Energy")
    ax1.set_title("Energy Cost")
    ax1.grid(axis="y", alpha=0.25)
    for i, v in enumerate(energy):
        ax1.text(i, v + energy_std[i] + 25, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    ax2.bar(x, mass, yerr=mass_std, color=colors, alpha=0.92, capsize=6)
    ax2.axhline(400.0, color="#d62828", linestyle="--", linewidth=1.4, label="Target 400 t")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Final Mass (t)")
    ax2.set_title("Final Dry Mass")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right")
    for i, v in enumerate(mass):
        ax2.text(i, v + mass_std[i] + 2.5, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    width = 0.24
    ax3.bar(x - width, zero_unsafe * 100.0, width=width, color="#2a9d8f", label="Zero Unsafe")
    ax3.bar(x, completion * 100.0, width=width, color="#457b9d", label="Completion")
    ax3.bar(x + width, inband * 100.0, width=width, color="#e9c46a", label="In-Band")
    ax3.set_xticks(x, labels)
    ax3.set_ylabel("Rate (%)")
    ax3.set_ylim(0, 110)
    ax3.set_title("Constraint Satisfaction")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(loc="upper right")

    for i, label in enumerate(labels):
        ax4.scatter(energy[i], mass[i], s=180, color=colors[i], edgecolors="black", linewidths=0.8)
        ax4.annotate(
            f"{label}\nE={energy[i]:.1f}\nM={mass[i]:.1f}",
            (energy[i], mass[i]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
        )
    ax4.axhline(400.0, color="#d62828", linestyle="--", linewidth=1.2)
    ax4.set_xlabel("Energy")
    ax4.set_ylabel("Final Mass (t)")
    ax4.set_title("Energy-Mass Tradeoff")
    ax4.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = DEBUG_ROOT / "action_space_current_env_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "action_space_current_env_compare_showcase.png"
    csv_path = out_dir / "action_space_current_env_compare_summary.csv"

    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    csv_lines = [
        "mode,mass_mean,mass_std,energy_mean,energy_std,zero_unsafe_rate,completion_rate,inband_rate"
    ]
    for label in labels:
        item = summary[label]
        csv_lines.append(
            f"{label},{item['mass_mean']},{item['mass_std']},{item['energy_mean']},{item['energy_std']},"
            f"{item['zero_unsafe_rate']},{item['completion_rate']},{item['inband_rate']}"
        )
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    print(f"Saved figure to: {png_path}")
    print(f"Saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
