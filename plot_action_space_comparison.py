from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from algorithms.discrete_ddqn import DiscreteDDQNAgent
from algorithms.hybrid_td3 import HybridTD3Agent
from algorithms.td3 import TD3Agent
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "action_space_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModeSpec:
    mode: str
    run_name: str
    checkpoint_relpath: str = "checkpoints/best_model.pth"
    target_mass: float = 400.0
    decision_interval: int = 5
    max_steps: int = 288
    use_gru_encoder: bool = True
    gru_hidden_dim: int = 96


@dataclass
class EvalSummary:
    mode: str
    run_name: str
    checkpoint: str
    mean_reward: float
    std_reward: float
    mean_mass: float
    std_mass: float
    mean_energy: float
    std_energy: float
    safe_rate: float
    inband_rate: float
    qualified_rate: float
    mean_target_band_distance: float


MODE_SPECS = [
    ModeSpec(mode="DD", run_name="dd_gru_serial_manual_001"),
    ModeSpec(mode="CD", run_name="cd_gru_serial_manual_001"),
    ModeSpec(mode="CC", run_name="cc_gru_serial_manual_001"),
]


def _target_band_distance(mass: float, low: float, high: float) -> float:
    if mass < low:
        return low - mass
    if mass > high:
        return mass - high
    return 0.0


def _evaluate_checkpoint(spec: ModeSpec, seeds: Iterable[int]) -> EvalSummary:
    run_dir = RUNS_ROOT / spec.run_name
    checkpoint = run_dir / spec.checkpoint_relpath
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    reward_list = []
    mass_list = []
    energy_list = []
    safe_flags = []
    inband_flags = []
    qualified_flags = []
    band_distance_list = []

    for seed in seeds:
        env = ThickenerDewateringEnv(
            max_steps=spec.max_steps,
            decision_interval=spec.decision_interval,
            target_mass=spec.target_mass,
            mode=spec.mode,
            reward_config=RewardConfig(target_mass=spec.target_mass, max_steps=spec.max_steps),
        )
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        if spec.mode == "DD":
            agent = DiscreteDDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_gru_encoder=spec.use_gru_encoder,
                gru_hidden_dim=spec.gru_hidden_dim,
            )
        elif spec.mode == "CD":
            agent = HybridTD3Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                q_uf_low=float(env.action_space.low[0]),
                q_uf_high=float(env.action_space.high[0]),
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_gru_encoder=spec.use_gru_encoder,
                gru_hidden_dim=spec.gru_hidden_dim,
            )
        else:
            agent = TD3Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_low=env.action_space.low,
                action_high=env.action_space.high,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_gru_encoder=spec.use_gru_encoder,
                gru_hidden_dim=spec.gru_hidden_dim,
            )
        agent.load(str(checkpoint))

        state, _ = env.reset(seed=int(seed))
        done = False
        total_reward = 0.0
        unsafe_episode = False
        final_info = None

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            unsafe_episode = unsafe_episode or bool(info.get("safety_violation", False))
            final_info = info
            state = next_state
            done = terminated or truncated

        if final_info is None:
            raise RuntimeError(f"No rollout info collected for {spec.mode} seed={seed}")

        final_mass = float(final_info.get("current_mass", 0.0))
        total_energy = float(final_info.get("total_energy_cost", 0.0))
        low = spec.target_mass
        high = spec.target_mass + 20.0
        in_band = low <= final_mass <= high
        qualified = (not unsafe_episode) and in_band

        reward_list.append(total_reward)
        mass_list.append(final_mass)
        energy_list.append(total_energy)
        safe_flags.append(0.0 if unsafe_episode else 1.0)
        inband_flags.append(1.0 if in_band else 0.0)
        qualified_flags.append(1.0 if qualified else 0.0)
        band_distance_list.append(_target_band_distance(final_mass, low, high))

    return EvalSummary(
        mode=spec.mode,
        run_name=spec.run_name,
        checkpoint=str(checkpoint),
        mean_reward=float(np.mean(reward_list)),
        std_reward=float(np.std(reward_list)),
        mean_mass=float(np.mean(mass_list)),
        std_mass=float(np.std(mass_list)),
        mean_energy=float(np.mean(energy_list)),
        std_energy=float(np.std(energy_list)),
        safe_rate=float(np.mean(safe_flags)),
        inband_rate=float(np.mean(inband_flags)),
        qualified_rate=float(np.mean(qualified_flags)),
        mean_target_band_distance=float(np.mean(band_distance_list)),
    )


def _save_csv(path: Path, summaries: list[EvalSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for item in summaries:
            writer.writerow(asdict(item))


def _save_json(path: Path, summaries: list[EvalSummary], seeds: list[int]) -> None:
    payload = {
        "seeds": seeds,
        "summaries": [asdict(item) for item in summaries],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_figure(path: Path, summaries: list[EvalSummary]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [item.mode for item in summaries]
    mean_mass = [item.mean_mass for item in summaries]
    mean_energy = [item.mean_energy for item in summaries]
    mean_reward = [item.mean_reward for item in summaries]
    safe_rate = [item.safe_rate * 100.0 for item in summaries]
    inband_rate = [item.inband_rate * 100.0 for item in summaries]
    qualified_rate = [item.qualified_rate * 100.0 for item in summaries]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_mass = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_rate = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    colors = ["#6B8E23", "#D98E04", "#2C7FB8"]
    x = np.arange(len(modes))

    bars = ax_mass.bar(x, mean_mass, color=colors, width=0.6)
    ax_mass.axhspan(400, 420, color="#9FD3C7", alpha=0.25, label="Target Band 400-420 t")
    ax_mass.set_xticks(x, modes)
    ax_mass.set_ylabel("Final Dry Mass (t)")
    ax_mass.set_title("(a) Dry Mass Comparison")
    ax_mass.legend(loc="upper left")
    for bar, value in zip(bars, mean_mass):
        ax_mass.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    bars = ax_energy.bar(x, mean_energy, color=colors, width=0.6)
    ax_energy.set_xticks(x, modes)
    ax_energy.set_ylabel("Total Energy Cost")
    ax_energy.set_title("(b) Energy Cost Comparison")
    for bar, value in zip(bars, mean_energy):
        ax_energy.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    width = 0.22
    ax_rate.bar(x - width, safe_rate, width=width, color="#4CAF50", label="Safe Rate")
    ax_rate.bar(x, inband_rate, width=width, color="#FFC107", label="In-Band Rate")
    ax_rate.bar(x + width, qualified_rate, width=width, color="#E64A19", label="Qualified Rate")
    ax_rate.set_xticks(x, modes)
    ax_rate.set_ylim(0, 105)
    ax_rate.set_ylabel("Rate (%)")
    ax_rate.set_title("(c) Safety and Task Success")
    ax_rate.legend(loc="upper right")

    ax_table.axis("off")
    cell_text = []
    for item in summaries:
        cell_text.append(
            [
                item.mode,
                f"{item.mean_reward:.1f}",
                f"{item.mean_mass:.1f}",
                f"{item.mean_energy:.1f}",
                f"{item.safe_rate*100:.0f}%",
                f"{item.inband_rate*100:.0f}%",
                f"{item.qualified_rate*100:.0f}%",
            ]
        )
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Mode", "Reward", "Mass (t)", "Energy", "Safe", "In-Band", "Qualified"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.7)
    ax_table.set_title("(d) Numerical Summary")

    fig.suptitle(
        "Action-Space Comparison Under the Same TD3-GRU Curriculum\n"
        "Deterministic evaluation of the best available checkpoints for DD / CD / CC",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    seeds = [42, 43, 44, 45, 46]
    summaries = [_evaluate_checkpoint(spec, seeds) for spec in MODE_SPECS]
    csv_path = OUTPUT_DIR / "action_space_comparison_summary.csv"
    json_path = OUTPUT_DIR / "action_space_comparison_summary.json"
    fig_path = OUTPUT_DIR / "action_space_comparison_f_style.png"
    _save_csv(csv_path, summaries)
    _save_json(json_path, summaries, seeds)
    _save_figure(fig_path, summaries)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved FIG: {fig_path}")


if __name__ == "__main__":
    main()
