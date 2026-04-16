"""
Rule-based control baselines for the thickener dewatering project.

Purpose:
- Test whether the current project can be controlled successfully without RL.
- Compare "safety-first" and "target-first" heuristic controllers under the same environment.

Usage:
    python rule_based_baselines.py
    python rule_based_baselines.py --seeds 10 --seed_start 91
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets


TARGET_MASS = 400.0
TARGET_HIGH = 415.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rule-based baselines")
    parser.add_argument("--seeds", type=int, default=5, help="Number of evaluation seeds")
    parser.add_argument("--seed_start", type=int, default=91, help="Starting seed")
    parser.add_argument("--steps", type=int, default=288, help="Decision steps per episode")
    parser.add_argument("--interval", type=int, default=5, help="Physical minutes per decision step")
    parser.add_argument("--target", type=float, default=TARGET_MASS, help="Target dry mass")
    parser.add_argument(
        "--save_json",
        type=str,
        default="logs/rule_based_baselines.json",
        help="Where to save summary JSON",
    )
    return parser.parse_args()


def make_env(seed: int, steps: int, interval: int, target: float) -> Tuple[ThickenerDewateringEnv, np.ndarray]:
    reward_config = RewardConfig(target_mass=target, max_steps=steps)
    env = ThickenerDewateringEnv(
        max_steps=steps,
        decision_interval=interval,
        target_mass=target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        verbose=False,
    )
    obs, _ = env.reset(seed=seed)
    return env, obs


def simulate_candidate(
    env: ThickenerDewateringEnv, action: np.ndarray
) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    rng_state = np.random.get_state()
    try:
        sim_env = copy.deepcopy(env)
        result = sim_env.step(action)
    finally:
        np.random.set_state(rng_state)
    return result


@dataclass
class ControllerContext:
    env: ThickenerDewateringEnv
    obs: np.ndarray
    info_history: List[Dict]
    action_history: List[np.ndarray]


class BaseController:
    name = "base"

    def reset(self):
        pass

    def act(self, ctx: ControllerContext) -> np.ndarray:
        raise NotImplementedError


class ConstantController(BaseController):
    def __init__(self, name: str, q_uf: float, q_fp: float):
        self.name = name
        self.q_uf = float(q_uf)
        self.q_fp = float(q_fp)

    def act(self, ctx: ControllerContext) -> np.ndarray:
        del ctx
        return np.array([self.q_uf, self.q_fp], dtype=np.float32)


class HysteresisController(BaseController):
    def __init__(self, low_q: float, high_q: float, low_th: float, high_th: float, q_fp: float):
        self.name = "hysteresis_cuf"
        self.low_q = float(low_q)
        self.high_q = float(high_q)
        self.low_th = float(low_th)
        self.high_th = float(high_th)
        self.q_fp = float(q_fp)
        self.mode = "low"

    def reset(self):
        self.mode = "low"

    def act(self, ctx: ControllerContext) -> np.ndarray:
        c_uf = float(ctx.obs[0])
        if self.mode == "low" and c_uf >= self.high_th:
            self.mode = "high"
        elif self.mode == "high" and c_uf <= self.low_th:
            self.mode = "low"
        q_uf = self.high_q if self.mode == "high" else self.low_q
        return np.array([q_uf, self.q_fp], dtype=np.float32)


class SafetyFirstLookaheadController(BaseController):
    def __init__(self):
        self.name = "lookahead_safety_first"
        self.candidates = [
            (q_uf, q_fp)
            for q_uf in [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
            for q_fp in [0, 5, 10, 15, 20, 30, 40, 50, 60, 70]
        ]

    def act(self, ctx: ControllerContext) -> np.ndarray:
        best_safe = None
        best_safe_score = float("inf")
        best_fallback = None
        best_fallback_score = float("inf")

        for q_uf, q_fp in self.candidates:
            action = np.array([q_uf, q_fp], dtype=np.float32)
            sim_obs, _, _, _, sim_info = simulate_candidate(ctx.env, action)

            v_buf = float(sim_obs[1])
            c_uf = float(sim_obs[0])
            delta = float(sim_info["delta_m_fp"])
            energy = float(sim_info["energy_cost_step"])
            safety_score = (
                delta
                + 0.5 * energy
                + 20.0 * max(v_buf - 18.0, 0.0)
                + 100.0 * max(v_buf - 26.0, 0.0)
                + 200.0 * max(c_uf - 0.745, 0.0)
            )

            if sim_info["safety_violation"]:
                if safety_score < best_fallback_score:
                    best_fallback_score = safety_score
                    best_fallback = action
            else:
                if safety_score < best_safe_score:
                    best_safe_score = safety_score
                    best_safe = action

        return best_safe if best_safe is not None else best_fallback


class TargetTrackingLookaheadController(BaseController):
    def __init__(self):
        self.name = "lookahead_target_tracking"
        self.candidates = [
            (0, 0),
            (0, 20),
            (0, 40),
            (5, 20),
            (8, 20),
            (10, 20),
            (12, 30),
            (14, 40),
            (16, 50),
            (18, 50),
            (20, 50),
        ]

    def act(self, ctx: ControllerContext) -> np.ndarray:
        best_action = None
        best_score = float("inf")

        for q_uf, q_fp in self.candidates:
            action = np.array([q_uf, q_fp], dtype=np.float32)
            sim_obs, _, _, _, sim_info = simulate_candidate(ctx.env, action)

            current_mass = float(sim_info["current_mass"])
            v_buf = float(sim_obs[1])
            c_uf = float(sim_obs[0])
            remaining = max(ctx.env.max_steps - ctx.env.policy_stepcnt - 1, 1)
            desired_rate = max(ctx.env.reward_config.target_mass - current_mass, 0.0) / remaining
            delta = float(sim_info["delta_m_fp"])

            score = 0.0
            if sim_info["safety_violation"]:
                score += 1e6
            score += 800.0 * max(v_buf - 28.0, 0.0)
            score += 800.0 * max(c_uf - 0.745, 0.0)
            score += 10.0 * abs(delta - desired_rate)
            score += 2.0 * float(sim_info["energy_cost_step"])
            score += 5.0 * max(current_mass - TARGET_HIGH, 0.0)

            if score < best_score:
                best_score = score
                best_action = action

        return best_action


def run_controller(
    controller: BaseController,
    seed: int,
    steps: int,
    interval: int,
    target: float,
) -> Dict[str, float]:
    env, obs = make_env(seed, steps, interval, target)
    controller.reset()

    done = False
    total_reward = 0.0
    info_history: List[Dict] = []
    action_history: List[np.ndarray] = []
    info = {}

    while not done:
        ctx = ControllerContext(
            env=env,
            obs=obs,
            info_history=info_history,
            action_history=action_history,
        )
        action = controller.act(ctx)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        info_history.append(info)
        action_history.append(action.copy())
        done = terminated or truncated

    unsafe_steps = sum(1 for item in info_history if item.get("safety_violation", False))
    final_mass = float(info.get("current_mass", 0.0))
    energy_cost = float(info.get("total_energy_cost", 0.0))
    in_band = TARGET_MASS <= final_mass <= TARGET_HIGH

    return {
        "final_mass": final_mass,
        "energy_cost": energy_cost,
        "unsafe_steps": float(unsafe_steps),
        "reward": float(total_reward),
        "in_target_band": float(in_band),
        "safe_and_in_band": float(in_band and unsafe_steps == 0),
        "zero_unsafe": float(unsafe_steps == 0),
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    masses = np.array([r["final_mass"] for r in results], dtype=float)
    energy = np.array([r["energy_cost"] for r in results], dtype=float)
    unsafe_steps = np.array([r["unsafe_steps"] for r in results], dtype=float)
    rewards = np.array([r["reward"] for r in results], dtype=float)
    return {
        "n": int(len(results)),
        "mass_mean": float(np.mean(masses)),
        "mass_std": float(np.std(masses)),
        "energy_mean": float(np.mean(energy)),
        "unsafe_steps_mean": float(np.mean(unsafe_steps)),
        "reward_mean": float(np.mean(rewards)),
        "target_band_rate": float(np.mean([r["in_target_band"] for r in results])),
        "zero_unsafe_rate": float(np.mean([r["zero_unsafe"] for r in results])),
        "safe_target_rate": float(np.mean([r["safe_and_in_band"] for r in results])),
    }


def main():
    args = parse_args()
    seeds = [args.seed_start + i for i in range(args.seeds)]

    controllers: List[BaseController] = [
        ConstantController("constant_low_targetish", q_uf=8.0, q_fp=20.0),
        ConstantController("constant_safe_high", q_uf=18.0, q_fp=50.0),
        HysteresisController(low_q=8.0, high_q=18.0, low_th=0.70, high_th=0.74, q_fp=50.0),
        SafetyFirstLookaheadController(),
        TargetTrackingLookaheadController(),
    ]

    summary = {}
    for controller in controllers:
        results = [
            run_controller(controller, seed, args.steps, args.interval, args.target)
            for seed in seeds
        ]
        summary[controller.name] = {
            "summary": summarize(results),
            "per_seed": {str(seed): result for seed, result in zip(seeds, results)},
        }

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Rule-based baseline results")
    print("=" * 80)
    for name, payload in summary.items():
        s = payload["summary"]
        print(
            f"{name:24s} | mass={s['mass_mean']:.1f}±{s['mass_std']:.1f} t"
            f" | unsafe={s['unsafe_steps_mean']:.1f}"
            f" | in_band={s['target_band_rate']:.1%}"
            f" | zero_unsafe={s['zero_unsafe_rate']:.1%}"
            f" | safe_target={s['safe_target_rate']:.1%}"
        )
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
