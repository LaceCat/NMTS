"""
Run PID baselines on the thickener dewatering environment.

This script provides a classical-control baseline that can be compared against
DD/CD/CC reinforcement-learning policies under the same environment and target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from algorithms.pid_controller import (
    DualLoopPIDController,
    PIDControllerConfig,
    PurePIDController,
    PurePIDControllerConfig,
)
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from utils.metrics import compute_action_variation, evaluate_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PID baseline controllers")
    parser.add_argument(
        "--controller",
        type=str,
        default="engineering",
        choices=["engineering", "pure"],
        help="PID controller variant",
    )
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument("--seeds", type=int, default=5, help="Number of evaluation seeds")
    parser.add_argument("--seed_start", type=int, default=91, help="Starting seed")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps per episode")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Physical minutes per decision step")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument(
        "--save_json",
        type=str,
        default="logs/pid_baseline_eval.json",
        help="Path for JSON summary",
    )
    return parser.parse_args()


def make_env(seed: int, steps: int, interval: int, target: float, mode: str) -> ThickenerDewateringEnv:
    reward_config = RewardConfig(target_mass=target, max_steps=steps)
    env = ThickenerDewateringEnv(
        max_steps=steps,
        decision_interval=interval,
        target_mass=target,
        mode=mode,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )
    env.reset(seed=seed)
    return env


def run_pid(seed: int, steps: int, interval: int, target: float, mode: str, controller_kind: str) -> Dict[str, float]:
    env = make_env(seed, steps, interval, target, mode)
    obs, _ = env.reset(seed=seed)
    if controller_kind == "pure":
        controller = PurePIDController(
            PurePIDControllerConfig(),
            mode=mode,
        )
    else:
        controller = DualLoopPIDController(
            PIDControllerConfig(
                target_mass=target,
                target_mass_high=target + 20.0,
                total_steps=steps,
            ),
            mode=mode,
        )

    done = False
    total_reward = 0.0
    info_history: List[Dict] = []
    action_history: List[np.ndarray] = []
    applied_action_history: List[np.ndarray] = []

    while not done:
        action = controller.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        info_history.append(info)
        action_history.append(action.copy())
        applied_action_history.append(
            np.array(
                [
                    float(info.get("applied_q_uf", 0.0)),
                    float(info.get("applied_q_fp", 0.0)),
                ],
                dtype=np.float32,
            )
        )
        obs = next_obs
        done = bool(terminated or truncated)

    metrics = evaluate_episode(
        info_history,
        env.reward_config.target_mass_low,
        env.reward_config.target_mass_high,
    )
    unsafe_steps = int(sum(1 for item in info_history if item.get("safety_violation", False)))

    fp_switches = 0
    for i in range(1, len(applied_action_history)):
        prev_on = applied_action_history[i - 1][1] > 1e-6
        curr_on = applied_action_history[i][1] > 1e-6
        if prev_on != curr_on:
            fp_switches += 1

    return {
        "controller": controller.name,
        "reward": float(total_reward),
        "final_mass": float(metrics["final_mass"]),
        "energy_cost": float(metrics["energy_cost"]),
        "eei": float(metrics["eei"]),
        "safety_violations": float(metrics["safety_violations"]),
        "unsafe_steps": float(unsafe_steps),
        "completion": float(metrics["final_mass"] >= target),
        "inband": float(metrics["inband"]),
        "switch_count": float(fp_switches),
        "action_delta_mean": float(compute_action_variation(applied_action_history)),
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    def std(key: str) -> float:
        return float(np.std([r[key] for r in results]))

    return {
        "n": int(len(results)),
        "reward_mean": mean("reward"),
        "reward_std": std("reward"),
        "mass_mean": mean("final_mass"),
        "mass_std": std("final_mass"),
        "energy_mean": mean("energy_cost"),
        "energy_std": std("energy_cost"),
        "eei_mean": mean("eei"),
        "unsafe_steps_mean": mean("unsafe_steps"),
        "zero_unsafe_rate": float(np.mean([r["unsafe_steps"] == 0 for r in results])),
        "completion_rate": mean("completion"),
        "inband_rate": mean("inband"),
        "switch_count_mean": mean("switch_count"),
        "action_delta_mean": mean("action_delta_mean"),
    }


def main():
    args = parse_args()
    seeds = [args.seed_start + i for i in range(args.seeds)]
    results = [run_pid(seed, args.steps, args.interval, args.target, args.mode, args.controller) for seed in seeds]
    summary = summarize(results)

    payload = {
        "controller": args.controller,
        "mode": args.mode,
        "target": args.target,
        "steps": args.steps,
        "interval": args.interval,
        "summary": summary,
        "per_seed": {str(seed): result for seed, result in zip(seeds, results)},
    }

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 96)
    print(f"PID baseline evaluation | controller={args.controller} | mode={args.mode}")
    print("=" * 96)
    print(
        f"mass={summary['mass_mean']:.2f}±{summary['mass_std']:.2f} t"
        f" | energy={summary['energy_mean']:.2f}±{summary['energy_std']:.2f}"
        f" | eei={summary['eei_mean']:.2f}"
        f" | zero_unsafe={summary['zero_unsafe_rate']:.1%}"
        f" | completion={summary['completion_rate']:.1%}"
        f" | inband={summary['inband_rate']:.1%}"
        f" | switch={summary['switch_count_mean']:.1f}"
        f" | act_delta={summary['action_delta_mean']:.4f}"
    )
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
