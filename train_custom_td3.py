"""
Custom TD3 training entry with a dense handcrafted reward.

Goal:
- Slightly exceed 400 t at the end of the episode.
- Keep the process safe.
- Minimize energy cost after safety and production are satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from algorithms.td3 import TD3Agent
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets


class CustomRewardScheme:
    """
    Dense reward tailored for:
    - dry mass slightly above 400 t,
    - absolute safety preference,
    - lower energy cost.
    """

    def __init__(self, target_mass: float = 400.0, upper_mass: float = 420.0):
        self.target_mass = float(target_mass)
        self.upper_mass = float(upper_mass)

    def compute(
        self,
        delta_m_fp: float,
        energy_cost: float,
        is_safe: bool,
        current_mass: float,
        step: int,
        max_steps: int,
    ):
        reward = 0.0
        breakdown = {}

        if current_mass < self.target_mass:
            prod_reward = delta_m_fp * 3.0
        else:
            prod_reward = -delta_m_fp * 5.0
        reward += prod_reward
        breakdown["production"] = prod_reward

        energy_penalty = -energy_cost * 0.5
        reward += energy_penalty
        breakdown["energy"] = energy_penalty

        safety_penalty = 0.0
        if not is_safe:
            safety_penalty = -150.0
        reward += safety_penalty
        breakdown["safety"] = safety_penalty

        terminal_reward = 0.0
        done = False
        if step >= max_steps - 1:
            done = True
            if current_mass < self.target_mass:
                terminal_reward = -(self.target_mass - current_mass) * 10.0
            elif self.target_mass <= current_mass <= self.upper_mass:
                terminal_reward = 2000.0
            else:
                terminal_reward = -(current_mass - self.upper_mass) * 10.0

        reward += terminal_reward
        breakdown["terminal"] = terminal_reward
        breakdown["total"] = reward

        return float(reward), breakdown, done


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD3 with custom handcrafted reward")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--max_steps", type=int, default=288, help="Decision steps per episode")
    parser.add_argument("--interval", type=int, default=5, help="Physical minutes per decision step")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument("--upper_mass", type=float, default=420.0, help="Upper preferred mass bound")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Random warmup steps")
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument(
        "--exploration_noise",
        type=float,
        default=0.12,
        help="Exploration noise scale for TD3 (fraction of action range when <= 1).",
    )
    parser.add_argument(
        "--policy_noise",
        type=float,
        default=0.12,
        help="Target policy smoothing noise for TD3.",
    )
    parser.add_argument(
        "--noise_clip",
        type=float,
        default=0.20,
        help="Noise clip for TD3 target policy smoothing.",
    )
    parser.add_argument(
        "--policy_freq",
        type=int,
        default=3,
        help="Delayed policy update frequency for TD3.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/custom_td3",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument("--save_every", type=int, default=50, help="Checkpoint interval in episodes")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_custom_td3(args):
    set_seed(args.seed)

    reward_config = RewardConfig(target_mass=args.target, max_steps=args.max_steps)
    env = ThickenerDewateringEnv(
        max_steps=args.max_steps,
        decision_interval=args.interval,
        target_mass=args.target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )

    custom_reward = CustomRewardScheme(target_mass=args.target, upper_mass=args.upper_mass)

    agent = TD3Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        exploration_noise=args.exploration_noise,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        device=args.device,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "training_metrics.json"

    history = []
    total_steps = 0
    best_safe_mass_gap = float("inf")

    print("=== 开始基于自定义奖励的 TD3 浓密脱水训练 ===")
    print(
        f"device={args.device} | episodes={args.episodes} | max_steps={args.max_steps} | "
        f"interval={args.interval} | target={args.target}~{args.upper_mass}"
    )

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        episode_energy = 0.0
        safety_violations = 0
        final_breakdown = {}
        info = {}

        for _ in range(args.max_steps):
            if total_steps < args.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            next_state, _, env_done, env_truncated, info = env.step(action)
            reward, breakdown, custom_done = custom_reward.compute(
                delta_m_fp=info["delta_m_fp"],
                energy_cost=info["energy_cost_step"],
                is_safe=not info["safety_violation"],
                current_mass=info["current_mass"],
                step=info["policy_step"],
                max_steps=args.max_steps,
            )

            done = bool(env_done or env_truncated or custom_done)

            agent.store_transition(state, action, reward, next_state, done)
            if total_steps >= args.warmup_steps:
                agent.update()

            state = next_state
            episode_reward += reward
            episode_energy += info["energy_cost_step"]
            if info["safety_violation"]:
                safety_violations += 1
            total_steps += 1
            final_breakdown = breakdown

            if done:
                break

        final_mass = float(info.get("current_mass", 0.0))
        safe_episode = safety_violations == 0
        safe_mass_gap = abs(final_mass - args.target) if safe_episode else float("inf")

        history.append(
            {
                "episode": episode + 1,
                "reward": float(episode_reward),
                "final_mass": final_mass,
                "energy": float(episode_energy),
                "safety_violations": int(safety_violations),
                "safe_episode": safe_episode,
                "reward_breakdown": final_breakdown,
            }
        )

        print(
            f"回合: {episode + 1:3d} | 总奖励: {episode_reward:8.1f} | "
            f"最终干矿量: {final_mass:7.1f} t | 能耗: {episode_energy:7.1f} | "
            f"安全违规次数: {safety_violations:3d}"
        )

        if safe_mass_gap < best_safe_mass_gap:
            best_safe_mass_gap = safe_mass_gap
            agent.save(str(save_dir / "best_safe_target_model.pth"))

        if (episode + 1) % args.save_every == 0:
            agent.save(str(save_dir / f"custom_td3_ep{episode + 1}.pth"))
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

    agent.save(str(save_dir / "final_model.pth"))
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"训练结束，结果保存在: {save_dir}")


if __name__ == "__main__":
    train_custom_td3(parse_args())
