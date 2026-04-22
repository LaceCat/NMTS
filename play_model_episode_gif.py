from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from evaluate import _transform_obs_for_sac, build_agent_and_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full-episode GIF playback for a saved model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--algo", type=str, default="sac", choices=["esac", "sac", "td3"], help="Algorithm type")
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Minutes per decision step")
    parser.add_argument("--seed", type=int, default=91, help="Rollout seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--output_dir", type=str, default="plots/model_episode_gif", help="Output directory")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    parser.add_argument("--dpi", type=int, default=100, help="GIF render dpi")
    parser.add_argument("--uf_control_mode", type=str, default="absolute", choices=["absolute", "delta"])
    parser.add_argument("--uf_delta_max", type=float, default=5.0)
    parser.add_argument("--disable_post_target_fp_governor", action="store_true")
    parser.add_argument("--q_fp_delta_max", type=float, default=-1.0)
    parser.add_argument("--disable_low_buffer_fp_guard", action="store_true")
    parser.add_argument("--low_buffer_fp_threshold", type=float, default=0.0)
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


def extract_layers(env: ThickenerDewateringEnv) -> list[float]:
    return [float(max(0.0, env.thickener.d2c(layer_density / 1e6))) for layer_density in env.thickener_state]


def transform_state_if_needed(args: argparse.Namespace, state: np.ndarray, steps: int) -> np.ndarray:
    if args.algo.lower() == "sac":
        return _transform_obs_for_sac(state, args.target, steps)
    return state


def rollout(args: argparse.Namespace) -> tuple[list[dict], dict]:
    env = build_env(args)
    agent, needs_adapt, select_action_adapted = build_agent_and_adapter(args, env)
    if not agent.load(args.checkpoint):
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}")

    raw_state, _ = env.reset(seed=args.seed)
    state = transform_state_if_needed(args, raw_state, env.max_steps)

    records: list[dict] = [
        {
            "seed": int(args.seed),
            "decision_step": 0,
            "minute": 0,
            "m_fp": 0.0,
            "c_uf": float(raw_state[0]),
            "v_buf": float(raw_state[1]),
            "q_uf": 0.0,
            "q_fp_env": 0.0,
            "q_fp_actual": 0.0,
            "q_fp_cmd": 0.0,
            "qf": float(raw_state[7]),
            "cf": float(raw_state[8]),
            "price": float(raw_state[5]),
            "energy": 0.0,
            "reward": 0.0,
            "fp_busy": False,
            "layers": extract_layers(env),
        }
    ]

    done = False
    total_reward = 0.0
    while not done:
        if needs_adapt:
            action = select_action_adapted(state, deterministic=True)
        else:
            action = agent.select_action(state, deterministic=True)

        next_raw_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        records.append(
            {
                "seed": int(args.seed),
                "decision_step": int(info.get("policy_step", len(records))),
                "minute": int(env.timecnt),
                "m_fp": float(info.get("current_mass", 0.0)),
                "c_uf": float(next_raw_state[0]),
                "v_buf": float(next_raw_state[1]),
                "q_uf": float(info.get("applied_q_uf", action[0])),
                "q_fp_env": float(info.get("applied_q_fp", action[1])),
                "q_fp_actual": float(info.get("actual_q_fp", info.get("applied_q_fp", action[1]))),
                "q_fp_cmd": float(info.get("commanded_q_fp", action[1])),
                "qf": float(next_raw_state[7]),
                "cf": float(next_raw_state[8]),
                "price": float(next_raw_state[5]),
                "energy": float(info.get("total_energy_cost", 0.0)),
                "reward": float(reward),
                "fp_busy": bool(info.get("fp_busy", False)),
                "layers": extract_layers(env),
            }
        )
        total_reward += float(reward)
        state = transform_state_if_needed(args, next_raw_state, env.max_steps)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "algo": args.algo,
        "mode": args.mode,
        "seed": int(args.seed),
        "final_mass": float(records[-1]["m_fp"]),
        "final_energy": float(records[-1]["energy"]),
        "max_c_uf": float(max(r["c_uf"] for r in records)),
        "max_v_buf": float(max(r["v_buf"] for r in records)),
        "total_reward": float(total_reward),
        "steps": int(records[-1]["decision_step"]),
        "minutes": int(records[-1]["minute"]),
    }
    return records, summary


def create_animation(records: list[dict], run_name: str, target: float, output_path: Path, fps: int, dpi: int) -> None:
    minutes = np.array([r["minute"] for r in records], dtype=float)
    mass = np.array([r["m_fp"] for r in records], dtype=float)
    c_uf = np.array([r["c_uf"] for r in records], dtype=float)
    v_buf = np.array([r["v_buf"] for r in records], dtype=float)
    q_uf = np.array([r["q_uf"] for r in records], dtype=float)
    q_fp_env = np.array([r["q_fp_env"] for r in records], dtype=float)
    q_fp_actual = np.array([r["q_fp_actual"] for r in records], dtype=float)
    q_fp_cmd = np.array([r["q_fp_cmd"] for r in records], dtype=float)
    qf = np.array([r["qf"] for r in records], dtype=float)
    cf = np.array([r["cf"] for r in records], dtype=float)
    price = np.array([r["price"] for r in records], dtype=float)
    energy = np.array([r["energy"] for r in records], dtype=float)
    layers = np.array([r["layers"] for r in records], dtype=float)
    fp_busy = np.array([1.0 if r["fp_busy"] else 0.0 for r in records], dtype=float)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    ax_mass = fig.add_subplot(gs[0, 0])
    ax_cuf = fig.add_subplot(gs[0, 1])
    ax_vbuf = fig.add_subplot(gs[1, 0])
    ax_act = fig.add_subplot(gs[1, 1])
    ax_feed = fig.add_subplot(gs[2, 0])
    ax_price = fig.add_subplot(gs[2, 1])
    ax_tank = fig.add_subplot(gs[0:2, 2])
    ax_profile = fig.add_subplot(gs[2, 2])

    seed = records[0]["seed"]
    fig.suptitle(f"Episode Playback | {run_name} | seed={seed}", fontsize=13, fontweight="bold")

    x_max = max(float(minutes[-1]), 1.0)

    def setup(ax, title: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Minute", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)

    setup(ax_mass, "Cumulative Mass (t)", "Mass")
    ax_mass.axhspan(target * 0.95, target * 1.1, color="#d8f3dc", alpha=0.6)
    ax_mass.axhline(target, color="#2d6a4f", linestyle="--", linewidth=1)
    ax_mass.set_ylim(0, max(float(mass.max()) * 1.05, target * 1.3))

    setup(ax_cuf, "Underflow Concentration", "C_uf")
    ax_cuf.axhline(0.75, color="red", linestyle="--", linewidth=1)
    ax_cuf.set_ylim(0, max(0.85, float(c_uf.max()) * 1.1))

    setup(ax_vbuf, "Buffer Volume", "V_buf")
    ax_vbuf.axhline(30.0, color="red", linestyle="--", linewidth=1)
    ax_vbuf.set_ylim(0, max(33.0, float(v_buf.max()) * 1.1))

    setup(ax_act, "Control Outputs", "Flow (m^3/h)")
    ax_act.set_ylim(0, 75)

    setup(ax_feed, "Feed", "Qf")
    ax_feed.set_ylim(34, 51)
    ax_feed2 = ax_feed.twinx()
    ax_feed2.set_ylabel("Cf", fontsize=7)
    ax_feed2.set_ylim(0.28, 0.47)

    setup(ax_price, "Price and Energy", "Price")
    ax_price.set_ylim(0.35, 1.3)
    ax_price2 = ax_price.twinx()
    ax_price2.set_ylabel("Energy", fontsize=7)
    ax_price2.set_ylim(0, max(1.0, float(energy.max()) * 1.1))

    ax_tank.set_title("Thickener Layers", fontsize=9)
    ax_tank.set_xlim(-0.6, 0.6)
    ax_tank.set_ylim(0, 10)
    ax_tank.set_xticks([])
    ax_tank.set_yticks(np.arange(0.5, 10.5))
    ax_tank.set_yticklabels([f"L{10 - i}" for i in range(10)], fontsize=6)

    ax_profile.set_title("Layer Concentration Profile", fontsize=9)
    ax_profile.set_xlabel("Concentration", fontsize=7)
    ax_profile.set_ylabel("Layer", fontsize=7)
    ax_profile.set_xlim(0, max(0.85, float(layers.max()) * 1.1))
    ax_profile.set_ylim(0.5, 10.5)
    ax_profile.set_yticks(range(1, 11))
    ax_profile.grid(True, alpha=0.25)

    m_line, = ax_mass.plot([], [], color="#1f77b4", linewidth=2)
    m_dot, = ax_mass.plot([], [], "o", color="#0b3d91", ms=4)
    cuf_line, = ax_cuf.plot([], [], color="#6a4c93", linewidth=2)
    cuf_dot, = ax_cuf.plot([], [], "o", color="#3c096c", ms=4)
    v_line, = ax_vbuf.plot([], [], color="#2a9d8f", linewidth=2)
    v_dot, = ax_vbuf.plot([], [], "o", color="#1d7874", ms=4)
    quf_line, = ax_act.plot([], [], color="#f4a261", linewidth=2, label="Q_uf")
    qfp_cmd_line, = ax_act.plot([], [], color="#e76f51", linewidth=1.2, linestyle="--", label="Q_fp cmd")
    qfp_env_line, = ax_act.plot([], [], color="#ff6b6b", linewidth=1.2, linestyle=":", label="Q_fp env")
    qfp_actual_line, = ax_act.plot([], [], color="#e63946", linewidth=2, label="Q_fp actual")
    qf_line, = ax_feed.plot([], [], color="#457b9d", linewidth=2, label="Qf")
    cf_line, = ax_feed2.plot([], [], color="#1d3557", linewidth=2, label="Cf")
    p_line, = ax_price.plot([], [], color="#ff006e", linewidth=2, label="Price")
    e_line, = ax_price2.plot([], [], color="#fb8500", linewidth=2, label="Energy")

    busy_fill = ax_act.fill_between(minutes, 0, 70, where=fp_busy > 0.5, color="#ced4da", alpha=0.22, step="pre")
    _ = busy_fill
    cursors = [ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.4) for ax in [ax_mass, ax_cuf, ax_vbuf, ax_act, ax_feed, ax_price]]

    ax_act.legend(loc="upper left", fontsize=6)
    ax_feed.legend(loc="upper left", fontsize=6)
    ax_price.legend(loc="upper left", fontsize=6)

    cmap = plt.get_cmap("YlOrBr")
    rects = []
    txts = []
    for i in range(10):
        rect = plt.Rectangle((-0.35, i), 0.7, 1, facecolor=cmap(0), edgecolor="#555", linewidth=0.5)
        ax_tank.add_patch(rect)
        rects.append(rect)
        txt = ax_tank.text(0, i + 0.5, "", ha="center", va="center", fontsize=6)
        txts.append(txt)

    prof_line, = ax_profile.plot([], [], color="#8d0801", linewidth=2, marker="o", ms=3)
    ax_profile.axvline(0.75, color="red", linestyle="--", linewidth=0.8)

    info = fig.text(0.01, 0.02, "", fontsize=7, family="monospace", va="bottom")

    def update(frame: int):
        x = minutes[: frame + 1]
        m_line.set_data(x, mass[: frame + 1])
        m_dot.set_data([minutes[frame]], [mass[frame]])
        cuf_line.set_data(x, c_uf[: frame + 1])
        cuf_dot.set_data([minutes[frame]], [c_uf[frame]])
        v_line.set_data(x, v_buf[: frame + 1])
        v_dot.set_data([minutes[frame]], [v_buf[frame]])
        quf_line.set_data(x, q_uf[: frame + 1])
        qfp_cmd_line.set_data(x, q_fp_cmd[: frame + 1])
        qfp_env_line.set_data(x, q_fp_env[: frame + 1])
        qfp_actual_line.set_data(x, q_fp_actual[: frame + 1])
        qf_line.set_data(x, qf[: frame + 1])
        cf_line.set_data(x, cf[: frame + 1])
        p_line.set_data(x, price[: frame + 1])
        e_line.set_data(x, energy[: frame + 1])

        lv = layers[frame]
        for j in range(10):
            source_idx = 9 - j
            val = float(lv[source_idx])
            rects[j].set_facecolor(cmap(np.clip(val / 0.75, 0, 1)))
            txts[j].set_text(f"{val:.3f}")

        prof_line.set_data(lv, np.arange(1, 11))
        for cursor in cursors:
            cursor.set_xdata([minutes[frame], minutes[frame]])

        info.set_text(
            f"step={frame:3d}  min={records[frame]['minute']:4d}  "
            f"mass={records[frame]['m_fp']:.1f}t  C_uf={records[frame]['c_uf']:.4f}  "
            f"V_buf={records[frame]['v_buf']:.1f}\n"
            f"Q_uf={records[frame]['q_uf']:.1f}  Q_fp_cmd={records[frame]['q_fp_cmd']:.1f}  "
            f"Q_fp_env={records[frame]['q_fp_env']:.1f}  Q_fp_actual={records[frame]['q_fp_actual']:.1f}\n"
            f"Qf={records[frame]['qf']:.1f}  Cf={records[frame]['cf']:.3f}  "
            f"price={records[frame]['price']:.3f}  energy={records[frame]['energy']:.1f}"
        )

    anim = animation.FuncAnimation(fig, update, frames=len(records), interval=max(1, int(1000 / fps)), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records, summary = rollout(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(args.checkpoint).stem
    stem = f"{args.algo}_{args.mode.lower()}_{checkpoint_name}_seed{args.seed}"
    gif_path = output_dir / f"{stem}.gif"
    summary_path = output_dir / f"{stem}_summary.json"

    create_animation(records, checkpoint_name, args.target, gif_path, args.fps, args.dpi)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"GIF saved to: {gif_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
