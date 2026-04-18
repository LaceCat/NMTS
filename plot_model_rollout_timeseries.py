from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import build_agent_and_adapter
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full-rollout time-series for a saved model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"F:\毕设\claude-code\runs\cc_batch_dryrun_lowconc_probe_v2scan\checkpoints\selected_epoch9_balanced.pth",
        help="Checkpoint path. Defaults to the current selected CC continuous-control model.",
    )
    parser.add_argument("--algo", type=str, default="td3", choices=["esac", "td3"], help="Algorithm type")
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Minutes per decision step")
    parser.add_argument("--seed", type=int, default=91, help="Rollout seed")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="__agent_debug__/cc_rollout_timeseries",
        help="Directory for PNG/CSV/JSON outputs",
    )
    return parser.parse_args()


def rollout(args: argparse.Namespace):
    reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        mode=args.mode,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )

    agent, needs_adapt, select_action_adapted = build_agent_and_adapter(args, env)
    if not agent.load(args.checkpoint):
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}")

    obs, _ = env.reset(seed=args.seed)
    rows = []
    done = False

    while not done:
        if needs_adapt:
            action = select_action_adapted(obs, deterministic=True)
        else:
            action = agent.select_action(obs, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "decision_step": int(info["policy_step"]),
                "minute": int(env.timecnt),
                "c_uf": float(next_obs[0]),
                "v_buf": float(next_obs[1]),
                "c_aver": float(next_obs[2]),
                "m_fp": float(next_obs[3]),
                "price": float(next_obs[5]),
                "q_uf": float(info.get("applied_q_uf", action[0])),
                "q_fp": float(info.get("applied_q_fp", action[1])),
                "q_fp_cmd": float(info.get("commanded_q_fp", action[1])),
                "energy_cost_step": float(info.get("energy_cost_step", 0.0)),
                "total_energy_cost": float(info.get("total_energy_cost", 0.0)),
                "reward": float(reward),
                "safety_violation": bool(info.get("safety_violation", False)),
                "fp_busy": bool(info.get("fp_busy", False)),
                "fp_downtime_remain": int(info.get("fp_downtime_remain", 0)),
                "fp_cycle_mass": float(info.get("fp_cycle_mass", 0.0)),
                "fp_total_cycles": int(info.get("fp_total_cycles", 0)),
                "fp_batch_triggered": bool(info.get("fp_batch_triggered", False)),
            }
        )
        obs = next_obs
        done = bool(terminated or truncated)

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], target: float, output_path: Path) -> None:
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

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1, ax2, ax3 = axes

    ax1.plot(minutes, c_uf, label="C_uf", color="#6a4c93", linewidth=2.0)
    ax1.plot(minutes, c_aver, label="C_aver", color="#1d3557", linewidth=1.8, alpha=0.85)
    ax1.axhline(0.75, color="#d00000", linestyle="--", linewidth=1.0, label="C_uf limit")
    ax1.set_ylabel("Concentration")
    ax1.set_title("CC Model Full-Rollout Concentration")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(minutes, v_buf, label="V_buf", color="#2a9d8f", linewidth=2.0)
    ax2.axhline(30.0, color="#d00000", linestyle="--", linewidth=1.0, label="V_buf limit")
    ax2.set_ylabel("Buffer Volume (m^3)")
    ax2.set_title("Buffer Tank Volume")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    ax3.plot(minutes, q_uf, label="Q_uf", color="#f4a261", linewidth=2.0)
    ax3.plot(minutes, q_fp_cmd, label="Q_fp cmd", color="#e76f51", linewidth=1.4, linestyle="--", alpha=0.9)
    ax3.plot(minutes, q_fp, label="Q_fp applied", color="#e63946", linewidth=2.0)
    busy_mask = fp_busy > 0.5
    if np.any(busy_mask):
        ax3.fill_between(minutes, 0, 70, where=busy_mask, color="#ced4da", alpha=0.22, step="pre", label="FP busy")
    ax3.set_ylabel("Flow (m^3/h)")
    ax3.set_xlabel("Minute")
    ax3.set_title("Control Outputs")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    summary = (
        f"Final mass = {m_fp[-1]:.2f} t | Target = {target:.1f} t | "
        f"Total energy = {total_energy[-1]:.2f}"
    )
    fig.suptitle(summary, fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = rollout(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem
    stem = f"{args.mode.lower()}_{checkpoint_name}_seed{args.seed}"

    png_path = output_dir / f"{stem}_timeseries.png"
    csv_path = output_dir / f"{stem}_timeseries.csv"
    json_path = output_dir / f"{stem}_summary.json"

    save_plot(rows, args.target, png_path)
    save_csv(rows, csv_path)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": args.mode,
        "seed": args.seed,
        "final_mass": rows[-1]["m_fp"],
        "final_total_energy_cost": rows[-1]["total_energy_cost"],
        "max_c_uf": max(row["c_uf"] for row in rows),
        "max_v_buf": max(row["v_buf"] for row in rows),
        "unsafe_steps": int(sum(1 for row in rows if row["safety_violation"])),
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
