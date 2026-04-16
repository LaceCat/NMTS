from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets


def parse_args():
    parser = argparse.ArgumentParser(description="Run a constant-action sanity check on the physical environment.")
    parser.add_argument("--q_uf", type=float, default=25.0, help="Constant underflow pump action")
    parser.add_argument("--q_fp", type=float, default=35.0, help="Constant filter press pump action")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps to simulate")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Minutes per decision step")
    parser.add_argument("--target", type=float, default=400.0, help="Target mass used by env config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/sanity_constant_control",
        help="Directory for CSV/JSON/plot outputs",
    )
    return parser.parse_args()


def extract_layers(env: ThickenerDewateringEnv):
    return [
        float(max(0.0, env.thickener.d2c(layer_density / 1e6)))
        for layer_density in env.thickener_state
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )

    obs, _ = env.reset(seed=args.seed)
    action = np.array([args.q_uf, args.q_fp], dtype=np.float32)

    rows = []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        layers = extract_layers(env)
        rows.append(
            {
                "policy_step": int(info["policy_step"]),
                "minute": int(env.timecnt),
                "c_uf": float(obs[0]),
                "v_buf": float(obs[1]),
                "c_aver": float(obs[2]),
                "m_fp": float(obs[3]),
                "mass_buf": float(obs[4]),
                "price": float(obs[5]),
                "qf": float(obs[7]),
                "cf": float(obs[8]),
                "reward": float(reward),
                "energy_cost_step": float(info["energy_cost_step"]),
                "total_energy_cost": float(info["total_energy_cost"]),
                "safety_violation": bool(info["safety_violation"]),
                "safety_violations": int(info["safety_violations"]),
                "violations": " | ".join(info.get("violations", [])),
                **{f"layer_{index + 1}": float(value) for index, value in enumerate(layers)},
            }
        )
        done = bool(terminated or truncated)

    csv_path = output_dir / "constant_control.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    masses = [row["m_fp"] for row in rows]
    volumes = [row["v_buf"] for row in rows]
    cufs = [row["c_uf"] for row in rows]
    safety_flags = [row["safety_violation"] for row in rows]
    total_energy = [row["total_energy_cost"] for row in rows]
    steps = [row["policy_step"] for row in rows]

    summary = {
        "q_uf": args.q_uf,
        "q_fp": args.q_fp,
        "steps": len(rows),
        "final_mass": masses[-1],
        "final_v_buf": volumes[-1],
        "final_c_uf": cufs[-1],
        "total_energy_cost": total_energy[-1],
        "unsafe_steps": int(sum(1 for flag in safety_flags if flag)),
        "max_v_buf": max(volumes),
        "max_c_uf": max(cufs),
        "min_c_uf": min(cufs),
    }

    summary_path = output_dir / "constant_control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Constant Control Sanity Check | Q_uf={args.q_uf}, Q_fp={args.q_fp}")

    axes[0, 0].plot(steps, masses, color="#1f77b4")
    axes[0, 0].set_title("Cumulative Dry Mass")
    axes[0, 0].set_xlabel("Decision Step")
    axes[0, 0].set_ylabel("Mass (t)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(steps, volumes, color="#2a9d8f")
    axes[0, 1].axhline(30.0, color="#d00000", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Buffer Volume")
    axes[0, 1].set_xlabel("Decision Step")
    axes[0, 1].set_ylabel("V_buf (m^3)")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(steps, cufs, color="#6a4c93")
    axes[1, 0].axhline(0.75, color="#d00000", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Underflow Concentration")
    axes[1, 0].set_xlabel("Decision Step")
    axes[1, 0].set_ylabel("C_uf")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(steps, total_energy, color="#fb8500")
    axes[1, 1].set_title("Cumulative Energy Cost")
    axes[1, 1].set_xlabel("Decision Step")
    axes[1, 1].set_ylabel("Cost")
    axes[1, 1].grid(True, alpha=0.25)

    plt.tight_layout()
    plot_path = output_dir / "constant_control_overview.png"
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
