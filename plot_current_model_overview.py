from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from evaluate import _transform_obs_for_sac, build_agent_and_adapter


def _configure_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False


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
    parser.add_argument("--disable_post_target_idle_seeker", action="store_true")
    parser.add_argument("--disable_midcourse_quality_governor", action="store_true")
    parser.add_argument("--disable_late_concentration_keeper", action="store_true")
    parser.add_argument("--disable_late_target_compensator", action="store_true")
    parser.add_argument("--q_fp_delta_max", type=float, default=-1.0)
    parser.add_argument("--disable_low_buffer_fp_guard", action="store_true")
    parser.add_argument("--low_buffer_fp_threshold", type=float, default=0.0)
    parser.add_argument("--low_buffer_fp_max", type=float, default=0.0)
    parser.add_argument("--low_buffer_fp_guard_max_correction", type=float, default=2.0)
    parser.add_argument("--governor_total_correction_limit", type=float, default=-1.0)
    parser.add_argument("--direct_q_fp_physical_only", action="store_true")
    parser.add_argument("--enable_q_fp_actual_slew_limit", dest="enable_q_fp_actual_slew_limit", action="store_true")
    parser.add_argument("--disable_q_fp_actual_slew_limit", dest="enable_q_fp_actual_slew_limit", action="store_false")
    parser.add_argument("--q_fp_actual_slew_up_per_minute", type=float, default=9.0)
    parser.add_argument("--q_fp_actual_slew_down_per_minute", type=float, default=9.0)
    parser.add_argument("--q_fp_actual_slew_restart_vbuf_limit", type=float, default=2.5)
    parser.add_argument("--enable_q_fp_actual_prestop_taper", dest="enable_q_fp_actual_prestop_taper", action="store_true")
    parser.add_argument("--disable_q_fp_actual_prestop_taper", dest="enable_q_fp_actual_prestop_taper", action="store_false")
    parser.add_argument("--q_fp_actual_prestop_mass", type=float, default=1.5)
    parser.add_argument("--q_fp_actual_prestop_vbuf_limit", type=float, default=2.5)
    parser.add_argument("--q_fp_actual_prestop_floor_ratio", type=float, default=0.25)
    parser.add_argument("--enable_fp_edge_smoothing", action="store_true")
    parser.add_argument("--fp_edge_smoothing_steps", type=int, default=3)
    parser.add_argument("--fp_prestop_ramp_mass", type=float, default=2.0)
    parser.add_argument("--fp_prestop_ramp_floor", type=float, default=0.88)
    parser.add_argument("--fp_restart_ramp_minutes", type=int, default=3)
    parser.set_defaults(enable_q_fp_actual_slew_limit=True, enable_q_fp_actual_prestop_taper=False)
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
        enable_post_target_idle_seeker=not args.disable_post_target_idle_seeker,
        enable_midcourse_quality_governor=not args.disable_midcourse_quality_governor,
        enable_late_concentration_keeper=not args.disable_late_concentration_keeper,
        enable_late_target_compensator=not args.disable_late_target_compensator,
        q_fp_delta_max=args.q_fp_delta_max,
        direct_q_fp_physical_only=args.direct_q_fp_physical_only,
        enable_low_buffer_fp_guard=not args.disable_low_buffer_fp_guard,
        low_buffer_fp_threshold=args.low_buffer_fp_threshold,
        low_buffer_fp_max=args.low_buffer_fp_max,
        low_buffer_fp_guard_max_correction=args.low_buffer_fp_guard_max_correction,
        governor_total_correction_limit=args.governor_total_correction_limit,
        enable_fp_edge_smoothing=args.enable_fp_edge_smoothing,
        fp_edge_smoothing_steps=args.fp_edge_smoothing_steps,
        fp_prestop_ramp_mass=args.fp_prestop_ramp_mass,
        fp_prestop_ramp_floor=args.fp_prestop_ramp_floor,
        fp_restart_ramp_minutes=args.fp_restart_ramp_minutes,
        enable_q_fp_actual_slew_limit=args.enable_q_fp_actual_slew_limit,
        q_fp_actual_slew_up_per_minute=args.q_fp_actual_slew_up_per_minute,
        q_fp_actual_slew_down_per_minute=args.q_fp_actual_slew_down_per_minute,
        q_fp_actual_slew_restart_vbuf_limit=args.q_fp_actual_slew_restart_vbuf_limit,
        enable_q_fp_actual_prestop_taper=args.enable_q_fp_actual_prestop_taper,
        q_fp_actual_prestop_mass=args.q_fp_actual_prestop_mass,
        q_fp_actual_prestop_vbuf_limit=args.q_fp_actual_prestop_vbuf_limit,
        q_fp_actual_prestop_floor_ratio=args.q_fp_actual_prestop_floor_ratio,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )


def transform_state_if_needed(args: argparse.Namespace, state: np.ndarray, steps: int) -> np.ndarray:
    if args.algo.lower() == "sac":
        return _transform_obs_for_sac(state, args.target, steps)
    return state


def rollout(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    env = build_env(args)
    agent, needs_adapt, select_action_adapted = build_agent_and_adapter(args, env)
    if not agent.load(args.checkpoint):
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}")

    raw_obs, _ = env.reset(seed=args.seed)
    obs = transform_state_if_needed(args, raw_obs, env.max_steps)
    decision_rows: list[dict] = []
    minute_rows: list[dict] = []
    done = False

    while not done:
        if needs_adapt:
            action = select_action_adapted(obs, deterministic=True)
        else:
            action = agent.select_action(obs, deterministic=True)

        next_raw_obs, reward, terminated, truncated, info = env.step(action)
        decision_rows.append(
            {
                "decision_step": int(info["policy_step"]),
                "minute": int(env.timecnt),
                "energy_cost_step": float(info.get("energy_cost_step", 0.0)),
                "total_energy_cost": float(info.get("total_energy_cost", 0.0)),
                "reward": float(reward),
                "safety_violation": bool(info.get("safety_violation", False)),
                "dry_run_violation": bool(info.get("dry_run_violation", False)),
                "low_conc_violation": bool(info.get("low_conc_violation", False)),
                "midcourse_quality_governed": bool(info.get("midcourse_quality_governed", False)),
                "low_buffer_fp_guarded": bool(info.get("low_buffer_fp_guarded", False)),
                "late_target_compensated": bool(info.get("late_target_compensated", False)),
                "late_concentration_kept": bool(info.get("late_concentration_kept", False)),
                "post_target_idle_seeking": bool(info.get("post_target_idle_seeking", False)),
                "buffer_below_idle_threshold": bool(
                    info.get(
                        "buffer_below_idle_threshold",
                        not bool(info.get("buffer_running_now", False)),
                    )
                ),
                "buffer_running_now": bool(info.get("buffer_running_now", False)),
                "fp_busy": bool(info.get("fp_busy", False)),
                "q_fp_prestop_tapered": bool(info.get("q_fp_prestop_tapered", False)),
                "q_fp_slew_limited": bool(info.get("q_fp_slew_limited", False)),
                "fp_total_cycles": int(info.get("fp_total_cycles", 0)),
            }
        )
        mt = info.get("minute_trace_time", [])
        mq_uf = info.get("minute_trace_q_uf_actual", [])
        mq_fp = info.get("minute_trace_q_fp_actual", [])
        mc_uf = info.get("minute_trace_c_uf", [])
        mc_aver = info.get("minute_trace_c_aver", [])
        mv_buf = info.get("minute_trace_v_buf", [])
        mm_fp = info.get("minute_trace_m_fp", [])
        menergy = info.get("minute_trace_total_energy_cost", [])
        mprice = info.get("minute_trace_price", [])
        mbusy = info.get("minute_trace_fp_busy", [])
        for minute, q_uf_actual, q_fp_actual, c_uf, c_aver, v_buf, m_fp, total_energy_cost, price, fp_busy in zip(
            mt,
            mq_uf,
            mq_fp,
            mc_uf,
            mc_aver,
            mv_buf,
            mm_fp,
            menergy,
            mprice,
            mbusy,
        ):
            minute_rows.append(
                {
                    "minute": int(minute),
                    "c_uf": float(c_uf),
                    "c_aver": float(c_aver),
                    "v_buf": float(v_buf),
                    "m_fp": float(m_fp),
                    "price": float(price),
                    "q_uf_actual": float(q_uf_actual),
                    "q_fp_actual": float(q_fp_actual),
                    "total_energy_cost": float(total_energy_cost),
                    "fp_busy": bool(fp_busy),
                }
            )
        done = bool(terminated or truncated)
        obs = transform_state_if_needed(args, next_raw_obs, env.max_steps)

    return decision_rows, minute_rows


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(minute_rows: list[dict], decision_rows: list[dict], args: argparse.Namespace, output_path: Path) -> None:
    minutes = np.array([row["minute"] for row in minute_rows], dtype=float)
    c_uf = np.array([row["c_uf"] for row in minute_rows], dtype=float)
    c_aver = np.array([row["c_aver"] for row in minute_rows], dtype=float)
    v_buf = np.array([row["v_buf"] for row in minute_rows], dtype=float)
    q_uf_actual = np.array([row["q_uf_actual"] for row in minute_rows], dtype=float)
    q_fp_actual = np.array([row["q_fp_actual"] for row in minute_rows], dtype=float)
    m_fp = np.array([row["m_fp"] for row in minute_rows], dtype=float)
    total_energy = np.array([row["total_energy_cost"] for row in minute_rows], dtype=float)
    fp_busy = np.array([float(row["fp_busy"]) for row in minute_rows], dtype=float)
    price = np.array([row["price"] for row in minute_rows], dtype=float)
    buffer_below_idle = np.array([float(row["v_buf"] <= 1.5 + 1e-6) for row in minute_rows], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
    ax1, ax2, ax3, ax4, ax5 = axes

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
    ax2.axhline(1.5, color="#2d6a4f", linestyle=":", linewidth=1.0, label="Mixer power-off threshold")
    ax2.axhline(0.05, color="#84a98c", linestyle="--", linewidth=0.9, label="True empty threshold")
    idle_mask = buffer_below_idle > 0.5
    if np.any(idle_mask):
        ax2.fill_between(minutes, 0, 30, where=idle_mask, color="#d8f3dc", alpha=0.25, step="pre", label="Below idle threshold")
    ax2.set_ylabel("Buffer Volume (m^3)")
    ax2.set_title("Buffer Tank Volume")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    ax3.plot(minutes, q_uf_actual, label="Q_uf actual", color="#f4a261", linewidth=2.0)
    ax3.plot(minutes, q_fp_actual, label="Q_fp actual", color="#e63946", linewidth=2.1)
    busy_mask = fp_busy > 0.5
    if np.any(busy_mask):
        ax3.fill_between(minutes, 0, 70, where=busy_mask, color="#ced4da", alpha=0.22, step="pre", label="FP busy")
    ax3.set_ylabel("Flow (m^3/h)")
    ax3.set_title("Actual Control Outputs")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    ax4.plot(minutes, m_fp, label="Cumulative mass", color="#264653", linewidth=2.2)
    ax4.axhline(args.target, color="#2d6a4f", linestyle="--", linewidth=1.0, label="Target 400 t")
    ax4.plot(minutes, total_energy, label="Total energy", color="#fb8500", linewidth=1.8)
    ax4.set_ylabel("Mass / Energy")
    ax4.set_title("Task Completion and Energy")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best")

    ax5.step(minutes, price, where="post", label="Electricity price", color="#3a86ff", linewidth=2.0)
    ax5.fill_between(minutes, 0, price, step="post", color="#dbeafe", alpha=0.35)
    ax5.set_ylabel("Price")
    ax5.set_xlabel("Minute")
    ax5.set_title("Time-of-Use Electricity Price")
    ax5.grid(True, alpha=0.25)
    ax5.legend(loc="best")

    summary = (
        f"{args.algo.upper()}-{args.mode} | seed={args.seed} | "
        f"Final mass={m_fp[-1]:.2f} t | Total energy={total_energy[-1]:.2f} | "
        f"Unsafe={int(sum(row['safety_violation'] for row in decision_rows))} | "
        f"Dry-run={int(sum(row['dry_run_violation'] for row in decision_rows))}"
    )
    fig.suptitle(summary, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _configure_font()
    decision_rows, minute_rows = rollout(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem
    stem = f"{args.algo}_{args.mode.lower()}_{checkpoint_name}_seed{args.seed}"
    png_path = output_dir / f"{stem}_overview.png"
    csv_path = output_dir / f"{stem}_overview.csv"
    json_path = output_dir / f"{stem}_summary.json"

    save_plot(minute_rows, decision_rows, args, png_path)
    save_csv(minute_rows, csv_path)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "algo": args.algo,
        "mode": args.mode,
        "seed": args.seed,
        "enable_q_fp_actual_slew_limit": bool(args.enable_q_fp_actual_slew_limit),
        "q_fp_actual_slew_up_per_minute": float(args.q_fp_actual_slew_up_per_minute),
        "q_fp_actual_slew_down_per_minute": float(args.q_fp_actual_slew_down_per_minute),
        "q_fp_actual_slew_restart_vbuf_limit": float(args.q_fp_actual_slew_restart_vbuf_limit),
        "enable_q_fp_actual_prestop_taper": bool(args.enable_q_fp_actual_prestop_taper),
        "q_fp_actual_prestop_mass": float(args.q_fp_actual_prestop_mass),
        "q_fp_actual_prestop_vbuf_limit": float(args.q_fp_actual_prestop_vbuf_limit),
        "q_fp_actual_prestop_floor_ratio": float(args.q_fp_actual_prestop_floor_ratio),
        "enable_fp_edge_smoothing": bool(args.enable_fp_edge_smoothing),
        "fp_edge_smoothing_steps": int(args.fp_edge_smoothing_steps),
        "fp_prestop_ramp_mass": float(args.fp_prestop_ramp_mass),
        "fp_prestop_ramp_floor": float(args.fp_prestop_ramp_floor),
        "fp_restart_ramp_minutes": int(args.fp_restart_ramp_minutes),
        "final_mass": minute_rows[-1]["m_fp"],
        "final_total_energy_cost": minute_rows[-1]["total_energy_cost"],
        "max_c_uf": max(row["c_uf"] for row in minute_rows),
        "max_v_buf": max(row["v_buf"] for row in minute_rows),
        "max_q_uf_actual": max(row["q_uf_actual"] for row in minute_rows),
        "max_q_fp_actual": max(row["q_fp_actual"] for row in minute_rows),
        "mean_q_fp_actual_jump": float(np.mean(np.abs(np.diff([row["q_fp_actual"] for row in minute_rows])))) if len(minute_rows) > 1 else 0.0,
        "p95_q_fp_actual_jump": float(np.percentile(np.abs(np.diff([row["q_fp_actual"] for row in minute_rows])), 95)) if len(minute_rows) > 1 else 0.0,
        "max_q_fp_actual_jump": float(np.max(np.abs(np.diff([row["q_fp_actual"] for row in minute_rows])))) if len(minute_rows) > 1 else 0.0,
        "unsafe_steps": int(sum(1 for row in decision_rows if row["safety_violation"])),
        "dry_run_steps": int(sum(1 for row in decision_rows if row["dry_run_violation"])),
        "low_conc_steps": int(sum(1 for row in decision_rows if row["low_conc_violation"])),
        "q_fp_prestop_tapered_steps": int(sum(1 for row in decision_rows if row["q_fp_prestop_tapered"])),
        "q_fp_slew_limited_steps": int(sum(1 for row in decision_rows if row["q_fp_slew_limited"])),
        "midcourse_quality_governor_steps": int(sum(1 for row in decision_rows if row["midcourse_quality_governed"])),
        "low_buffer_guard_steps": int(sum(1 for row in decision_rows if row["low_buffer_fp_guarded"])),
        "late_target_compensator_steps": int(sum(1 for row in decision_rows if row["late_target_compensated"])),
        "late_concentration_keeper_steps": int(sum(1 for row in decision_rows if row["late_concentration_kept"])),
        "post_target_idle_seeker_steps": int(sum(1 for row in decision_rows if row["post_target_idle_seeking"])),
        "below_idle_threshold_steps": int(sum(1 for row in minute_rows if row["v_buf"] <= 1.5 + 1e-6)),
        "fp_total_cycles": int(decision_rows[-1].get("fp_total_cycles", 0)),
        "png": str(png_path),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PNG saved to: {png_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
