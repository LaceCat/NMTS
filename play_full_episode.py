from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.td3 import TD3Agent
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from train_custom_td3 import CustomRewardScheme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-episode playback for the current TD3 policy.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to latest best_safe_target_model.pth")
    parser.add_argument("--output_dir", type=str, default="plots/full_episode_playback", help="Directory for GIF/CSV/JSON outputs")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps per episode")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Physical minutes per decision step")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument("--upper_mass", type=float, default=420.0, help="Preferred upper mass bound")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=None, help="Use this exact seed. If omitted, auto-search is used.")
    parser.add_argument("--seed_start", type=int, default=42, help="Starting seed for automatic selection")
    parser.add_argument("--search_seeds", type=int, default=100, help="Number of seeds to evaluate for automatic selection")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    parser.add_argument("--dpi", type=int, default=120, help="GIF render dpi")
    return parser.parse_args()


def find_latest_checkpoint(root: Path) -> Path:
    best_candidates = list(root.rglob("best_safe_target_model.pth"))
    if best_candidates:
        return max(best_candidates, key=lambda path: path.stat().st_mtime)
    final_candidates = list(root.rglob("final_model.pth"))
    if final_candidates:
        return max(final_candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError("No checkpoint found under checkpoints/")


def build_env(max_steps: int, interval: int, target: float) -> ThickenerDewateringEnv:
    reward_config = RewardConfig(target_mass=target, max_steps=max_steps)
    return ThickenerDewateringEnv(
        max_steps=max_steps,
        decision_interval=interval,
        target_mass=target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )


def build_agent(env: ThickenerDewateringEnv, device: str) -> TD3Agent:
    return TD3Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )


def extract_layers(env: ThickenerDewateringEnv) -> List[float]:
    return [
        float(max(0.0, env.thickener.d2c(layer_density / 1e6)))
        for layer_density in env.thickener_state
    ]


def format_violations(violations: List[str]) -> List[str]:
    translated = []
    for violation in violations:
        if "底流浓度超限" in violation:
            translated.append("Underflow concentration above limit")
        elif "缓冲罐超容" in violation:
            translated.append("Buffer volume above limit")
        else:
            translated.append(violation.encode("ascii", errors="ignore").decode("ascii") or "Safety violation")
    return translated


def initial_record(state: np.ndarray, layers: List[float], minute: int) -> Dict[str, Any]:
    return {
        "decision_step": 0,
        "minute": int(minute),
        "c_uf": float(state[0]),
        "v_buf": float(state[1]),
        "c_aver": float(state[2]),
        "m_fp": float(state[3]),
        "mass_buf": float(state[4]),
        "price": float(state[5]),
        "remaining_steps": float(state[6]),
        "qf": float(state[7]),
        "cf": float(state[8]),
        "q_uf": 0.0,
        "q_fp": 0.0,
        "env_reward": 0.0,
        "custom_reward": 0.0,
        "delta_m_fp": 0.0,
        "energy_cost_step": 0.0,
        "total_energy_cost": 0.0,
        "safety_violation": False,
        "safety_violations": 0,
        "target_reached": False,
        "violations": [],
        "layers": layers,
    }


def run_episode(
    agent: TD3Agent,
    env: ThickenerDewateringEnv,
    custom_reward: CustomRewardScheme,
    seed: int,
) -> Dict[str, Any]:
    state, _ = env.reset(seed=seed)
    records: List[Dict[str, Any]] = [initial_record(state, extract_layers(env), env.timecnt)]

    done = False
    env_reward_sum = 0.0
    custom_reward_sum = 0.0
    unsafe_steps = 0

    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, env_reward, terminated, truncated, info = env.step(action)
        custom_step_reward, _, custom_done = custom_reward.compute(
            delta_m_fp=info["delta_m_fp"],
            energy_cost=info["energy_cost_step"],
            is_safe=not info["safety_violation"],
            current_mass=info["current_mass"],
            step=max(int(info["policy_step"]) - 1, 0),
            max_steps=env.max_steps,
        )

        record = {
            "decision_step": int(info["policy_step"]),
            "minute": int(env.timecnt),
            "c_uf": float(next_state[0]),
            "v_buf": float(next_state[1]),
            "c_aver": float(next_state[2]),
            "m_fp": float(next_state[3]),
            "mass_buf": float(next_state[4]),
            "price": float(next_state[5]),
            "remaining_steps": float(next_state[6]),
            "qf": float(next_state[7]),
            "cf": float(next_state[8]),
            "q_uf": float(action[0]),
            "q_fp": float(action[1]),
            "env_reward": float(env_reward),
            "custom_reward": float(custom_step_reward),
            "delta_m_fp": float(info["delta_m_fp"]),
            "energy_cost_step": float(info["energy_cost_step"]),
            "total_energy_cost": float(info["total_energy_cost"]),
            "safety_violation": bool(info["safety_violation"]),
            "safety_violations": int(info["safety_violations"]),
            "target_reached": bool(info["target_reached"]),
            "violations": format_violations(list(info.get("violations", []))),
            "layers": extract_layers(env),
        }
        records.append(record)

        env_reward_sum += float(env_reward)
        custom_reward_sum += float(custom_step_reward)
        if record["safety_violation"]:
            unsafe_steps += 1

        state = next_state
        done = bool(terminated or truncated or custom_done)

    final_record = records[-1]
    summary = {
        "seed": int(seed),
        "steps": int(final_record["decision_step"]),
        "minutes": int(final_record["minute"]),
        "final_mass": float(final_record["m_fp"]),
        "total_energy_cost": float(final_record["total_energy_cost"]),
        "unsafe_steps": int(unsafe_steps),
        "target_reached": bool(final_record["target_reached"]),
        "env_reward_sum": float(env_reward_sum),
        "custom_reward_sum": float(custom_reward_sum),
        "mass_gap_to_target": float(abs(final_record["m_fp"] - custom_reward.target_mass)),
    }
    return {"records": records, "summary": summary}


def score_episode(result: Dict[str, Any]) -> Tuple[float, ...]:
    summary = result["summary"]
    return (
        float(summary["unsafe_steps"] > 0),
        float(summary["unsafe_steps"]),
        float(summary["mass_gap_to_target"]),
        float(summary["total_energy_cost"]),
    )


def choose_episode(
    agent: TD3Agent,
    env: ThickenerDewateringEnv,
    custom_reward: CustomRewardScheme,
    seed: int | None,
    seed_start: int,
    search_seeds: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    if seed is not None:
        result = run_episode(agent, env, custom_reward, seed)
        candidates.append(result["summary"])
        return result, candidates

    best_result = None
    best_score = None
    for candidate_seed in range(seed_start, seed_start + search_seeds):
        result = run_episode(agent, env, custom_reward, candidate_seed)
        candidates.append(result["summary"])
        current_score = score_episode(result)
        if best_result is None or current_score < best_score:
            best_result = result
            best_score = current_score
    return best_result, candidates


def save_csv(records: List[Dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = [
        "decision_step",
        "minute",
        "c_uf",
        "v_buf",
        "c_aver",
        "m_fp",
        "mass_buf",
        "price",
        "remaining_steps",
        "qf",
        "cf",
        "q_uf",
        "q_fp",
        "env_reward",
        "custom_reward",
        "delta_m_fp",
        "energy_cost_step",
        "total_energy_cost",
        "safety_violation",
        "safety_violations",
        "target_reached",
        "violations",
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "layer_5",
        "layer_6",
        "layer_7",
        "layer_8",
        "layer_9",
        "layer_10",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["violations"] = " | ".join(row["violations"])
            for layer_index, layer_value in enumerate(row.pop("layers"), start=1):
                row[f"layer_{layer_index}"] = layer_value
            writer.writerow(row)


def create_animation(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    checkpoint: Path,
    target: float,
    upper_mass: float,
    output_gif: Path,
    fps: int,
    dpi: int,
) -> None:
    minutes = np.array([record["minute"] for record in records], dtype=np.float32)
    mass = np.array([record["m_fp"] for record in records], dtype=np.float32)
    c_uf = np.array([record["c_uf"] for record in records], dtype=np.float32)
    v_buf = np.array([record["v_buf"] for record in records], dtype=np.float32)
    q_uf = np.array([record["q_uf"] for record in records], dtype=np.float32)
    q_fp = np.array([record["q_fp"] for record in records], dtype=np.float32)
    qf = np.array([record["qf"] for record in records], dtype=np.float32)
    cf = np.array([record["cf"] for record in records], dtype=np.float32)
    price = np.array([record["price"] for record in records], dtype=np.float32)
    total_energy = np.array([record["total_energy_cost"] for record in records], dtype=np.float32)
    env_reward = np.array([record["env_reward"] for record in records], dtype=np.float32)
    custom_reward = np.array([record["custom_reward"] for record in records], dtype=np.float32)
    unsafe = np.array([float(record["safety_violation"]) for record in records], dtype=np.float32)
    layers = np.array([record["layers"] for record in records], dtype=np.float32)

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[1.05, 1.05, 0.95],
        height_ratios=[1.0, 1.0, 1.0, 0.72],
        hspace=0.42,
        wspace=0.28,
    )

    ax_mass = fig.add_subplot(gs[0, 0])
    ax_cuf = fig.add_subplot(gs[0, 1])
    ax_vbuf = fig.add_subplot(gs[1, 0])
    ax_actions = fig.add_subplot(gs[1, 1])
    ax_feed = fig.add_subplot(gs[2, 0])
    ax_price = fig.add_subplot(gs[2, 1])
    ax_tank = fig.add_subplot(gs[0:2, 2])
    ax_profile = fig.add_subplot(gs[2, 2])
    ax_text = fig.add_subplot(gs[3, :])

    fig.suptitle(
        f"Full Episode Playback | {checkpoint.parent.name}",
        fontsize=15,
        fontweight="bold",
    )

    x_max = max(float(minutes[-1]), 1.0)

    def setup_axis(ax, title: str, ylabel: str):
        ax.set_title(title)
        ax.set_xlabel("Minute")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.25)

    setup_axis(ax_mass, "Cumulative Dry Mass", "Mass (t)")
    ax_mass.axhspan(target, upper_mass, color="#d8f3dc", alpha=0.8, label="Target band")
    ax_mass.axhline(target, color="#2d6a4f", linestyle="--", linewidth=1.2)
    ax_mass.axhline(upper_mass, color="#40916c", linestyle="--", linewidth=1.2)
    ax_mass.set_ylim(0, max(float(mass.max()) * 1.08, upper_mass * 1.10))

    setup_axis(ax_cuf, "Underflow Concentration", "C_uf")
    ax_cuf.axhline(0.75, color="#d00000", linestyle="--", linewidth=1.2, label="Safety limit")
    ax_cuf.set_ylim(0, max(0.85, float(c_uf.max()) * 1.08))

    setup_axis(ax_vbuf, "Buffer Volume", "V_buf (m^3)")
    ax_vbuf.axhline(30.0, color="#d00000", linestyle="--", linewidth=1.2, label="Safety limit")
    ax_vbuf.set_ylim(0, max(33.0, float(v_buf.max()) * 1.08))

    setup_axis(ax_actions, "Control Actions", "Flow (m^3/h)")
    ax_actions.set_ylim(0, 75.0)

    setup_axis(ax_feed, "Feed Disturbance", "Qf (m^3/h)")
    ax_feed.set_ylim(34.0, 51.0)
    ax_feed_twin = ax_feed.twinx()
    ax_feed_twin.set_ylabel("Cf")
    ax_feed_twin.set_ylim(0.28, 0.47)

    setup_axis(ax_price, "Price and Cumulative Energy", "Price")
    ax_price.set_ylim(0.35, 1.30)
    ax_price_twin = ax_price.twinx()
    ax_price_twin.set_ylabel("Energy Cost")
    ax_price_twin.set_ylim(0, max(1.0, float(total_energy.max()) * 1.10))

    ax_tank.set_title("Thickener 10-Layer State")
    ax_tank.set_xlim(-0.8, 0.8)
    ax_tank.set_ylim(0.0, 10.0)
    ax_tank.set_xticks([])
    ax_tank.set_yticks(np.arange(0.5, 10.5, 1.0))
    ax_tank.set_yticklabels([f"L{10 - index}" for index in range(10)])
    ax_tank.set_ylabel("Layer")
    ax_tank.grid(False)

    ax_profile.set_title("10-Layer Concentration Profile")
    ax_profile.set_xlabel("Concentration")
    ax_profile.set_ylabel("Layer index")
    ax_profile.set_xlim(0.0, max(0.85, float(layers.max()) * 1.08))
    ax_profile.set_ylim(0.5, 10.5)
    ax_profile.set_yticks(np.arange(1, 11))
    ax_profile.grid(True, alpha=0.25)

    ax_text.axis("off")

    unsafe_minutes = minutes[unsafe > 0.5]
    unsafe_cuf = c_uf[unsafe > 0.5]
    unsafe_vbuf = v_buf[unsafe > 0.5]
    if len(unsafe_minutes) > 0:
        ax_cuf.scatter(unsafe_minutes, unsafe_cuf, s=14, color="#d00000", alpha=0.65, label="Unsafe step")
        ax_vbuf.scatter(unsafe_minutes, unsafe_vbuf, s=14, color="#d00000", alpha=0.65, label="Unsafe step")

    mass_line, = ax_mass.plot([], [], color="#1f77b4", linewidth=2.2, label="M_FP")
    mass_dot, = ax_mass.plot([], [], "o", color="#0b3d91", markersize=5)
    cuf_line, = ax_cuf.plot([], [], color="#6a4c93", linewidth=2.0, label="C_uf")
    cuf_dot, = ax_cuf.plot([], [], "o", color="#3c096c", markersize=5)
    vbuf_line, = ax_vbuf.plot([], [], color="#2a9d8f", linewidth=2.0, label="V_buf")
    vbuf_dot, = ax_vbuf.plot([], [], "o", color="#1d7874", markersize=5)
    quf_line, = ax_actions.plot([], [], color="#f4a261", linewidth=2.0, label="Q_uf")
    qfp_line, = ax_actions.plot([], [], color="#e63946", linewidth=2.0, label="Q_fp")
    qf_line, = ax_feed.plot([], [], color="#457b9d", linewidth=2.0, label="Qf")
    cf_line, = ax_feed_twin.plot([], [], color="#1d3557", linewidth=2.0, label="Cf")
    price_line, = ax_price.plot([], [], color="#ff006e", linewidth=2.0, label="Price")
    energy_line, = ax_price_twin.plot([], [], color="#fb8500", linewidth=2.0, label="Cum energy")

    cursor_axes = [ax_mass, ax_cuf, ax_vbuf, ax_actions, ax_feed, ax_price]
    cursors = [ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.45) for ax in cursor_axes]

    ax_mass.legend(loc="upper left", fontsize=8)
    ax_cuf.legend(loc="upper left", fontsize=8)
    ax_vbuf.legend(loc="upper left", fontsize=8)
    ax_actions.legend(loc="upper left", fontsize=8)
    feed_handles = [qf_line, cf_line]
    ax_feed.legend(feed_handles, [handle.get_label() for handle in feed_handles], loc="upper left", fontsize=8)
    price_handles = [price_line, energy_line]
    ax_price.legend(price_handles, [handle.get_label() for handle in price_handles], loc="upper left", fontsize=8)

    tank_cmap = plt.get_cmap("YlOrBr")
    layer_rectangles = []
    layer_value_texts = []
    for layer_idx in range(10):
        rect = plt.Rectangle(
            (-0.45, layer_idx),
            0.90,
            1.0,
            facecolor=tank_cmap(0.0),
            edgecolor="#555555",
            linewidth=0.6,
        )
        ax_tank.add_patch(rect)
        layer_rectangles.append(rect)
        layer_text = ax_tank.text(
            0.0,
            layer_idx + 0.5,
            "",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )
        layer_value_texts.append(layer_text)
    ax_tank.plot([-0.45, -0.45], [0, 10], color="black", linewidth=1.4)
    ax_tank.plot([0.45, 0.45], [0, 10], color="black", linewidth=1.4)
    ax_tank.plot([-0.45, 0.45], [0, 0], color="black", linewidth=1.4)

    profile_line, = ax_profile.plot([], [], color="#8d0801", linewidth=2.2, marker="o", markersize=4)
    profile_limit = ax_profile.axvline(0.75, color="#d00000", linestyle="--", linewidth=1.0, label="Safety limit")
    ax_profile.legend(loc="lower right", fontsize=8)

    summary_text = ax_text.text(
        0.01,
        0.92,
        "",
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )

    def format_text(index: int) -> str:
        record = records[index]
        unsafe_so_far = int(np.sum(unsafe[: index + 1]))
        violation_text = ", ".join(record["violations"]) if record["violations"] else "-"
        layer_snapshot = record["layers"]
        return "\n".join(
            [
                "Current Frame  |  Full Process Snapshot",
                f"seed={summary['seed']}  step={record['decision_step']:03d}  minute={record['minute']:04d}  "
                f"mass={record['m_fp']:.2f} t  C_uf={record['c_uf']:.4f}  V_buf={record['v_buf']:.2f} m^3",
                f"Q_uf={record['q_uf']:.2f}  Q_fp={record['q_fp']:.2f}  "
                f"Qf={record['qf']:.2f}  Cf={record['cf']:.3f}  price={record['price']:.3f}  "
                f"cum_energy={record['total_energy_cost']:.2f}",
                f"env_reward={record['env_reward']:.2f}  custom_reward={record['custom_reward']:.2f}  "
                f"unsafe_now={bool(record['safety_violation'])}  unsafe_so_far={unsafe_so_far}  "
                f"target_reached={bool(record['target_reached'])}",
                f"layers(top->bottom)={', '.join(f'{value:.3f}' for value in layer_snapshot)}",
                f"violations={violation_text}",
                "",
                f"Episode Summary  |  final_mass={summary['final_mass']:.2f} t  "
                f"unsafe_steps={summary['unsafe_steps']}  total_energy={summary['total_energy_cost']:.2f}  "
                f"gap_to_400={summary['mass_gap_to_target']:.2f} t  "
                f"env_sum={summary['env_reward_sum']:.2f}  custom_sum={summary['custom_reward_sum']:.2f}",
            ]
        )

    def update(frame_index: int):
        x = minutes[: frame_index + 1]

        mass_line.set_data(x, mass[: frame_index + 1])
        mass_dot.set_data([minutes[frame_index]], [mass[frame_index]])
        cuf_line.set_data(x, c_uf[: frame_index + 1])
        cuf_dot.set_data([minutes[frame_index]], [c_uf[frame_index]])
        vbuf_line.set_data(x, v_buf[: frame_index + 1])
        vbuf_dot.set_data([minutes[frame_index]], [v_buf[frame_index]])
        quf_line.set_data(x, q_uf[: frame_index + 1])
        qfp_line.set_data(x, q_fp[: frame_index + 1])
        qf_line.set_data(x, qf[: frame_index + 1])
        cf_line.set_data(x, cf[: frame_index + 1])
        price_line.set_data(x, price[: frame_index + 1])
        energy_line.set_data(x, total_energy[: frame_index + 1])

        layer_values = layers[frame_index]
        for draw_index, rect in enumerate(layer_rectangles):
            source_index = 9 - draw_index
            layer_value = float(layer_values[source_index])
            normalized = np.clip(layer_value / 0.75, 0.0, 1.0)
            rect.set_facecolor(tank_cmap(normalized))
            layer_value_texts[draw_index].set_text(f"{layer_value:.3f}")
        profile_line.set_data(layer_values, np.arange(1, 11))

        for cursor in cursors:
            cursor.set_xdata([minutes[frame_index], minutes[frame_index]])

        ax_text.set_facecolor("#fff1f2" if unsafe[frame_index] > 0.5 else "#f8f9fa")
        summary_text.set_text(format_text(frame_index))

        return [
            mass_line,
            mass_dot,
            cuf_line,
            cuf_dot,
            vbuf_line,
            vbuf_dot,
            quf_line,
            qfp_line,
            qf_line,
            cf_line,
            price_line,
            energy_line,
            profile_line,
            profile_limit,
            summary_text,
            *cursors,
            *layer_rectangles,
            *layer_value_texts,
        ]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(records),
        interval=max(1, int(1000 / max(fps, 1))),
        blit=False,
    )
    anim.save(str(output_gif), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(repo_root / "checkpoints")
    checkpoint = checkpoint.resolve()

    env = build_env(args.max_steps, args.interval, args.target)
    agent = build_agent(env, args.device)
    if not agent.load(str(checkpoint)):
        raise RuntimeError(f"Failed to load checkpoint: {checkpoint}")

    custom_reward = CustomRewardScheme(target_mass=args.target, upper_mass=args.upper_mass)

    selected_result, candidates = choose_episode(
        agent=agent,
        env=env,
        custom_reward=custom_reward,
        seed=args.seed,
        seed_start=args.seed_start,
        search_seeds=args.search_seeds,
    )

    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = checkpoint.parent.name
    seed = selected_result["summary"]["seed"]
    stem = f"{run_name}_seed{seed}"

    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_summary.json"
    search_path = output_dir / f"{stem}_seed_search.json"
    gif_path = output_dir / f"{stem}.gif"

    save_csv(selected_result["records"], csv_path)

    payload = {
        "checkpoint": str(checkpoint),
        "selected_summary": selected_result["summary"],
        "target_mass": float(args.target),
        "upper_mass": float(args.upper_mass),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    search_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    create_animation(
        records=selected_result["records"],
        summary=selected_result["summary"],
        checkpoint=checkpoint,
        target=args.target,
        upper_mass=args.upper_mass,
        output_gif=gif_path,
        fps=args.fps,
        dpi=args.dpi,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Selected seed: {seed}")
    print(f"GIF saved to: {gif_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")
    print(f"Seed search saved to: {search_path}")


if __name__ == "__main__":
    main()
