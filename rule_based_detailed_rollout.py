from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from rule_based_quick_eval import (
    BaseController,
    ConstantController,
    ControllerContext,
    PriceTaperController,
    SafetyBiasedController,
    SimpleTaperController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a detailed rule-based rollout and export plots/data.")
    parser.add_argument(
        "--controller",
        type=str,
        default="simple_taper",
        choices=[
            "constant_10p5_20",
            "constant_10p75_20",
            "constant_11p0_20",
            "price_taper",
            "simple_taper",
            "safety_biased",
        ],
        help="Controller to run",
    )
    parser.add_argument("--seed", type=int, default=91, help="Random seed")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Minutes per decision step")
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/rule_rollouts",
        help="Parent directory for CSV/JSON/plot outputs",
    )
    return parser.parse_args()


def build_env(steps: int, interval: int, target: float, mode: str) -> ThickenerDewateringEnv:
    reward_config = RewardConfig(target_mass=target, max_steps=steps)
    return ThickenerDewateringEnv(
        max_steps=steps,
        decision_interval=interval,
        target_mass=target,
        mode=mode,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )


def build_controller(name: str) -> BaseController:
    if name == "constant_10p5_20":
        return ConstantController(name, 10.5, 20.0)
    if name == "constant_10p75_20":
        return ConstantController(name, 10.75, 20.0)
    if name == "constant_11p0_20":
        return ConstantController(name, 11.0, 20.0)
    if name == "price_taper":
        return PriceTaperController()
    if name == "simple_taper":
        return SimpleTaperController()
    if name == "safety_biased":
        return SafetyBiasedController()
    raise ValueError(f"Unsupported controller: {name}")


def extract_layers(env: ThickenerDewateringEnv) -> List[float]:
    return [
        float(max(0.0, env.thickener.d2c(layer_density / 1e6)))
        for layer_density in env.thickener_state
    ]


def run_rollout(controller: BaseController, seed: int, steps: int, interval: int, target: float, mode: str) -> List[Dict[str, float]]:
    env = build_env(steps, interval, target, mode)
    obs, _ = env.reset(seed=seed)
    controller.reset()

    rows: List[Dict[str, float]] = []
    info_history: List[Dict] = []
    action_history: List[np.ndarray] = []
    done = False

    while not done:
        ctx = ControllerContext(obs=obs, info_history=info_history, action_history=action_history)
        action = controller.act(ctx)
        next_obs, reward, terminated, truncated, info = env.step(action)
        layers = extract_layers(env)
        row: Dict[str, float] = {
            "policy_step": int(info["policy_step"]),
            "minute": int(env.timecnt),
            "c_uf": float(next_obs[0]),
            "v_buf": float(next_obs[1]),
            "c_aver": float(next_obs[2]),
            "m_fp": float(next_obs[3]),
            "mass_buf": float(next_obs[4]),
            "price": float(next_obs[5]),
            "remaining_steps": float(next_obs[6]),
            "qf": float(next_obs[7]),
            "cf": float(next_obs[8]),
            "q_uf": float(action[0]),
            "q_fp": float(action[1]),
            "reward": float(reward),
            "energy_cost_step": float(info["energy_cost_step"]),
            "total_energy_cost": float(info["total_energy_cost"]),
            "delta_m_fp": float(info["delta_m_fp"]),
            "safety_violation": float(bool(info["safety_violation"])),
            "safety_violations": float(info["safety_violations"]),
            "target_reached": float(bool(info["target_reached"])),
        }
        for idx, value in enumerate(layers, start=1):
            row[f"layer_{idx}"] = float(value)
        rows.append(row)

        info_history.append(info)
        action_history.append(action.copy())
        obs = next_obs
        done = bool(terminated or truncated)

    return rows


def build_summary(controller_name: str, seed: int, rows: List[Dict[str, float]], target: float) -> Dict[str, float]:
    final = rows[-1]
    final_mass = float(final["m_fp"])
    unsafe_steps = int(sum(1 for row in rows if row["safety_violation"] > 0.5))
    summary = {
        "controller": controller_name,
        "seed": int(seed),
        "steps": int(len(rows)),
        "minutes": int(final["minute"]),
        "final_mass": final_mass,
        "mass_gap_to_400": float(final_mass - target),
        "in_target_band": bool(400.0 <= final_mass <= 420.0),
        "unsafe_steps": unsafe_steps,
        "zero_unsafe": unsafe_steps == 0,
        "safe_target": bool(unsafe_steps == 0 and 400.0 <= final_mass <= 420.0),
        "total_energy_cost": float(final["total_energy_cost"]),
        "max_v_buf": float(max(row["v_buf"] for row in rows)),
        "max_c_uf": float(max(row["c_uf"] for row in rows)),
        "min_c_uf": float(min(row["c_uf"] for row in rows)),
    }
    return summary


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: List[Dict[str, float]], summary: Dict[str, float], path: Path) -> None:
    steps = [row["policy_step"] for row in rows]
    masses = [row["m_fp"] for row in rows]
    cufs = [row["c_uf"] for row in rows]
    volumes = [row["v_buf"] for row in rows]
    quf = [row["q_uf"] for row in rows]
    qfp = [row["q_fp"] for row in rows]
    prices = [row["price"] for row in rows]
    energy = [row["total_energy_cost"] for row in rows]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"Rule rollout | {summary['controller']} | seed={summary['seed']} | "
        f"mass={summary['final_mass']:.1f} t | energy={summary['total_energy_cost']:.1f} | unsafe={summary['unsafe_steps']}",
        fontsize=14,
    )

    axes[0, 0].plot(steps, masses, color="#1f77b4", linewidth=2.0)
    axes[0, 0].axhline(400.0, color="#2a9d8f", linestyle="--", linewidth=1.0)
    axes[0, 0].axhline(420.0, color="#2a9d8f", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Cumulative Dry Mass")
    axes[0, 0].set_xlabel("Decision Step")
    axes[0, 0].set_ylabel("Mass (t)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(steps, cufs, color="#6a4c93", linewidth=2.0)
    axes[0, 1].axhline(0.75, color="#d00000", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Underflow Concentration")
    axes[0, 1].set_xlabel("Decision Step")
    axes[0, 1].set_ylabel("C_uf")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(steps, volumes, color="#2a9d8f", linewidth=2.0)
    axes[1, 0].axhline(30.0, color="#d00000", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Buffer Volume")
    axes[1, 0].set_xlabel("Decision Step")
    axes[1, 0].set_ylabel("V_buf (m^3)")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(steps, quf, color="#f4a261", linewidth=2.0, label="Q_uf")
    axes[1, 1].plot(steps, qfp, color="#e63946", linewidth=2.0, label="Q_fp")
    axes[1, 1].set_title("Control Actions")
    axes[1, 1].set_xlabel("Decision Step")
    axes[1, 1].set_ylabel("Flow")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.25)

    axes[2, 0].plot(steps, prices, color="#ff006e", linewidth=2.0, label="Price")
    axes[2, 0].set_title("Electricity Price")
    axes[2, 0].set_xlabel("Decision Step")
    axes[2, 0].set_ylabel("Price")
    axes[2, 0].grid(True, alpha=0.25)

    axes[2, 1].plot(steps, energy, color="#fb8500", linewidth=2.0)
    axes[2, 1].set_title("Cumulative Energy Cost")
    axes[2, 1].set_xlabel("Decision Step")
    axes[2, 1].set_ylabel("Cost")
    axes[2, 1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    controller = build_controller(args.controller)
    rows = run_rollout(controller, args.seed, args.steps, args.interval, args.target, args.mode)
    summary = build_summary(controller.name, args.seed, rows, args.target)
    summary["mode"] = args.mode

    output_dir = Path(args.output_dir) / f"{controller.name}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "timeseries.csv"
    json_path = output_dir / "summary.json"
    plot_path = output_dir / "overview.png"

    save_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_plot(rows, summary, plot_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
