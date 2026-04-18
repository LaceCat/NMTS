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
from evaluate import _transform_obs_for_sac, build_agent_and_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a single static overview figure for the current model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--algo", type=str, default="sac", choices=["esac", "sac", "td3"])
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"])
    parser.add_argument("--target", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS)
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="plots/current_model_overview")
    parser.add_argument("--uf_control_mode", type=str, default="absolute", choices=["absolute", "delta"])
    parser.add_argument("--uf_delta_max", type=float, default=5.0)
    parser.add_argument("--disable_post_target_fp_governor", action="store_true")
    parser.add_argument("--q_fp_delta_max", type=float, default=-1.0)
    parser.add_argument("--disable_low_buffer_fp_guard", action="store_true")
    parser.add_argument("--low_buffer_fp_threshold", type=float, default=0.8)
    parser.add_argument("--low_buffer_fp_max", type=float, default=0.0)
    args = parser.parse_args()
    if args.algo.lower() == "sac":
        if "--uf_control_mode" not in __import__("sys").argv:
            args.uf_control_mode = "delta"
        if "--q_fp_delta_max" not in __import__("sys").argv:
            args.q_fp_delta_max = 12.0
    return args


def build_env(args: argparse.Namespace) -> ThickenerDewateringEnv:
    reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    return ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        mode=args.mode,
        uf_control_mode=args.uf_control_mode,
        uf_delta_max=args.uf_delta_max,
        enable_post_target_fp_governor=not args.disable_post_target_fp_governor,
        q_fp_delta_max=args.q_fp_delta_max,
        enable_low_buffer_fp_guard=not args.disable_low_buffer_fp_guard,
        low_buffer_fp_threshold=args.low_buffer_fp_threshold,
        low_buffer_fp_max=args.low_buffer_fp_max,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )


def transform_state_if_needed(args: argparse.Namespace, state: np.ndarray, steps: int) -> np.ndarray:
    if args.algo.lower() == "sac":
        return _transform_obs_for_sac(state, args.target, steps)
    return state


def rollout(args: argparse.Namespace) -> list[dict]:
    env = build_env(args)
    agent, needs_adapt, select_action_adapted = build_agent_and_adapter(args, env)
    if not agent.load(args.checkpoint):
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}")

    raw_obs, _ = env.reset(seed=args.seed)
    obs = transform_state_if_needed(args, raw_obs, env.max_steps)
    rows: list[dict] = []
    done = False

    while not done:
        if needs_adapt:
            action = select_action_adapted(obs, deterministic=True)
        else:
            action = agent.select_action(obs, deterministic=True)

        next_raw_obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "decision_step": int(info["policy_step"]),
                "minute": int(env.timecnt),
                "c_uf": float(next_raw_obs[0]),
                "v_buf": float(next_raw_obs[1]),
                "c_aver": float(next_raw_obs[2]),
                "m_fp": float(next_raw_obs[3]),
                "price": float(next_raw_obs[5]),
                "q_uf": float(info.get("applied_q_uf", action[0])),
                "q_fp": float(info.get("applied_q_fp", action[1])),
                "q_fp_cmd": float(info.get("commanded_q_fp", action[1])),
                "energy_cost_step": float(info.get("energy_cost_step", 0.0)),
                "total_energy_cost": float(info.get("total_energy_cost", 0.0)),
                "reward": float(reward),
                "safety_violation": bool(info.get("safety_violation", False)),
                "dry_run_violation": bool(info.get("dry_run_violation", False)),
                "low_conc_violation": bool(info.get("low_conc_violation", False)),
                "low_buffer_fp_guarded": bool(info.get("low_buffer_fp_guarded", False)),
                "fp_busy": bool(info.get("fp_busy", False)),
                "fp_total_cycles": int(info.get("fp_total_cycles", 0)),
            }
        )
        done = bool(terminated or truncated)
        obs = transform_state_if_needed(args, next_raw_obs, env.max_steps)

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], args: argparse.Namespace, output_path: Path) -> None:
    minutes = np.array([row["minute"] for row in rows], dtype=float)
    c_uf = np.array([row["c_uf"] for row in rows], dtype=float)
    c_aver = np.array([row["c_aver"] for row in rows], dtype=float)
    v_buf = np.array([row["v_buf"] for row in rows], dtype=float)
    q_uf = np.array([row["q_uf"] for row in rows], dtype=float)
    q_fp = np.array([row["q_fp"] for row in rows], dtype=float)
    q_fp_cmd = np.array([row["q_fp_cmd"] for row in rows], dtype=float)
    m_fp = np.array([row["m_fp"] for row in rows], dtype=float)
    total_energy = np.array([row["total_energy_cost"] for row in rows], dtype=float)
    fp_busy = np.array([float(row["fp_busy"]) for row in rows], dtype=float)
    low_buffer_guarded = np.array([float(row["low_buffer_fp_guarded"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(minutes, c_uf, label="C_uf", color="#6a4c93", linewidth=2.0)
    ax1.plot(minutes, c_aver, label="C_aver", color="#1d3557", linewidth=1.7, alpha=0.85)
    ax1.axhline(0.75, color="#d00000", linestyle="--", linewidth=1.0, label="C_uf limit")
    ax1.axhline(0.66, color="#f77f00", linestyle=":", linewidth=1.0, label="C_uf soft limit")
    ax1.set_ylabel("Concentration")
    ax1.set_title("Concentration Evolution")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(minutes, v_buf, label="V_buf", color="#2a9d8f", linewidth=2.0)
    ax2.axhline(30.0, color="#d00000", linestyle="--", linewidth=1.0, label="V_buf limit")
    ax2.set_ylabel("Buffer Volume (m^3)")
    ax2.set_title("Buffer Tank Volume")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    ax3.plot(minutes, q_uf, label="Q_uf", color="#f4a261", linewidth=2.0)
    ax3.plot(minutes, q_fp_cmd, label="Q_fp cmd", color="#e76f51", linewidth=1.3, linestyle="--", alpha=0.9)
    ax3.plot(minutes, q_fp, label="Q_fp applied", color="#e63946", linewidth=2.0)
    busy_mask = fp_busy > 0.5
    if np.any(busy_mask):
        ax3.fill_between(minutes, 0, 70, where=busy_mask, color="#ced4da", alpha=0.22, step="pre", label="FP busy")
    guard_mask = low_buffer_guarded > 0.5
    if np.any(guard_mask):
        ax3.fill_between(minutes, 0, 70, where=guard_mask, color="#fff3bf", alpha=0.28, step="pre", label="Low-buffer guard")
    ax3.set_ylabel("Flow (m^3/h)")
    ax3.set_title("Control Outputs")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    ax4.plot(minutes, m_fp, label="Cumulative mass", color="#264653", linewidth=2.2)
    ax4.axhline(args.target, color="#2d6a4f", linestyle="--", linewidth=1.0, label="Target 400 t")
    ax4.plot(minutes, total_energy, label="Total energy", color="#fb8500", linewidth=1.8)
    ax4.set_ylabel("Mass / Energy")
    ax4.set_xlabel("Minute")
    ax4.set_title("Task Completion and Energy")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best")

    summary = (
        f"{args.algo.upper()}-{args.mode} | seed={args.seed} | "
        f"Final mass={m_fp[-1]:.2f} t | Total energy={total_energy[-1]:.2f} | "
        f"Unsafe={int(sum(row['safety_violation'] for row in rows))} | "
        f"Dry-run={int(sum(row['dry_run_violation'] for row in rows))}"
    )
    fig.suptitle(summary, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = rollout(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem
    stem = f"{args.algo}_{args.mode.lower()}_{checkpoint_name}_seed{args.seed}"
    png_path = output_dir / f"{stem}_overview.png"
    csv_path = output_dir / f"{stem}_overview.csv"
    json_path = output_dir / f"{stem}_summary.json"

    save_plot(rows, args, png_path)
    save_csv(rows, csv_path)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "algo": args.algo,
        "mode": args.mode,
        "seed": args.seed,
        "final_mass": rows[-1]["m_fp"],
        "final_total_energy_cost": rows[-1]["total_energy_cost"],
        "max_c_uf": max(row["c_uf"] for row in rows),
        "max_v_buf": max(row["v_buf"] for row in rows),
        "unsafe_steps": int(sum(1 for row in rows if row["safety_violation"])),
        "dry_run_steps": int(sum(1 for row in rows if row["dry_run_violation"])),
        "low_conc_steps": int(sum(1 for row in rows if row["low_conc_violation"])),
        "low_buffer_guard_steps": int(sum(1 for row in rows if row["low_buffer_fp_guarded"])),
        "fp_total_cycles": int(rows[-1].get("fp_total_cycles", 0)),
        "png": str(png_path),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PNG saved to: {png_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
