from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets


def parse_args():
    parser = argparse.ArgumentParser(description="Quick rule-based evaluation on the thickener environment.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of evaluation seeds")
    parser.add_argument("--seed_start", type=int, default=91, help="Starting seed")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Minutes per decision step")
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument(
        "--save_json",
        type=str,
        default="__agent_debug__/rule_based_quick_eval.json",
        help="Path for summary JSON",
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


@dataclass
class ControllerContext:
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


class PriceTaperController(BaseController):
    name = "price_taper"

    def act(self, ctx: ControllerContext) -> np.ndarray:
        c_uf = float(ctx.obs[0])
        v_buf = float(ctx.obs[1])
        m_fp = float(ctx.obs[3])
        price = float(ctx.obs[5])

        if c_uf > 0.745 or v_buf > 5.0:
            return np.array([12.0, 20.0], dtype=np.float32)
        if m_fp >= 405.0:
            q_fp = 10.0 if v_buf > 0.05 else 0.0
            return np.array([0.0, q_fp], dtype=np.float32)
        if price <= 0.75:
            q_uf = 11.5 if c_uf < 0.73 else 11.0
            q_fp = 20.0
            return np.array([q_uf, q_fp], dtype=np.float32)
        if price >= 1.16:
            q_uf = 10.0
            q_fp = 20.0
            return np.array([q_uf, q_fp], dtype=np.float32)
        return np.array([10.75, 20.0], dtype=np.float32)


class SimpleTaperController(BaseController):
    name = "simple_taper"

    def act(self, ctx: ControllerContext) -> np.ndarray:
        c_uf = float(ctx.obs[0])
        v_buf = float(ctx.obs[1])
        m_fp = float(ctx.obs[3])

        if c_uf > 0.745 or v_buf > 5.0:
            return np.array([12.0, 20.0], dtype=np.float32)
        if m_fp < 320.0:
            return np.array([11.0, 20.0], dtype=np.float32)
        if m_fp < 390.0:
            return np.array([10.5, 20.0], dtype=np.float32)
        if m_fp < 400.0:
            return np.array([10.0, 20.0], dtype=np.float32)
        q_fp = 10.0 if v_buf > 0.05 else 0.0
        return np.array([0.0, q_fp], dtype=np.float32)


class SafetyBiasedController(BaseController):
    name = "safety_biased"

    def act(self, ctx: ControllerContext) -> np.ndarray:
        c_uf = float(ctx.obs[0])
        v_buf = float(ctx.obs[1])
        m_fp = float(ctx.obs[3])

        if c_uf > 0.74 or v_buf > 26.0:
            return np.array([12.0, 20.0], dtype=np.float32)
        if v_buf > 5.0:
            return np.array([11.5, 20.0], dtype=np.float32)
        if m_fp < 380.0:
            return np.array([10.75, 20.0], dtype=np.float32)
        if m_fp < 405.0:
            return np.array([10.25, 20.0], dtype=np.float32)
        q_fp = 10.0 if v_buf > 0.05 else 0.0
        return np.array([0.0, q_fp], dtype=np.float32)


def run_controller(controller: BaseController, seed: int, steps: int, interval: int, target: float, mode: str) -> Dict[str, float]:
    env = make_env(seed, steps, interval, target, mode)
    obs, _ = env.reset(seed=seed)
    controller.reset()

    done = False
    total_reward = 0.0
    info_history: List[Dict] = []
    action_history: List[np.ndarray] = []
    max_v_buf = float(obs[1])
    max_c_uf = float(obs[0])
    min_c_uf = float(obs[0])

    while not done:
        ctx = ControllerContext(
            obs=obs,
            info_history=info_history,
            action_history=action_history,
        )
        action = controller.act(ctx)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        info_history.append(info)
        action_history.append(action.copy())
        max_v_buf = max(max_v_buf, float(obs[1]))
        max_c_uf = max(max_c_uf, float(obs[0]))
        min_c_uf = min(min_c_uf, float(obs[0]))
        done = bool(terminated or truncated)

    unsafe_steps = sum(1 for item in info_history if item.get("safety_violation", False))
    final_mass = float(info.get("current_mass", 0.0))
    in_band = 400.0 <= final_mass <= 420.0
    zero_unsafe = unsafe_steps == 0

    return {
        "final_mass": final_mass,
        "energy_cost": float(info.get("total_energy_cost", 0.0)),
        "unsafe_steps": float(unsafe_steps),
        "reward": total_reward,
        "max_v_buf": max_v_buf,
        "max_c_uf": max_c_uf,
        "min_c_uf": min_c_uf,
        "in_target_band": float(in_band),
        "zero_unsafe": float(zero_unsafe),
        "safe_target": float(in_band and zero_unsafe),
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "n": int(len(results)),
        "mass_mean": float(np.mean([r["final_mass"] for r in results])),
        "mass_std": float(np.std([r["final_mass"] for r in results])),
        "energy_mean": float(np.mean([r["energy_cost"] for r in results])),
        "unsafe_mean": float(np.mean([r["unsafe_steps"] for r in results])),
        "reward_mean": float(np.mean([r["reward"] for r in results])),
        "max_v_buf_mean": float(np.mean([r["max_v_buf"] for r in results])),
        "max_c_uf_mean": float(np.mean([r["max_c_uf"] for r in results])),
        "target_band_rate": float(np.mean([r["in_target_band"] for r in results])),
        "zero_unsafe_rate": float(np.mean([r["zero_unsafe"] for r in results])),
        "safe_target_rate": float(np.mean([r["safe_target"] for r in results])),
    }


def main():
    args = parse_args()
    seeds = [args.seed_start + i for i in range(args.seeds)]

    controllers: List[BaseController] = [
        ConstantController("constant_10p5_20", 10.5, 20.0),
        ConstantController("constant_10p75_20", 10.75, 20.0),
        ConstantController("constant_11p0_20", 11.0, 20.0),
        PriceTaperController(),
        SimpleTaperController(),
        SafetyBiasedController(),
    ]

    summary: Dict[str, Dict] = {}
    for controller in controllers:
        results = [run_controller(controller, seed, args.steps, args.interval, args.target, args.mode) for seed in seeds]
        summary[controller.name] = {
            "summary": summarize(results),
            "per_seed": {str(seed): result for seed, result in zip(seeds, results)},
        }

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 96)
    print(f"Rule-based quick evaluation | mode={args.mode}")
    print("=" * 96)
    for name, payload in summary.items():
        s = payload["summary"]
        print(
            f"{name:18s} | mass={s['mass_mean']:7.1f}±{s['mass_std']:<6.1f} t"
            f" | energy={s['energy_mean']:7.1f}"
            f" | unsafe={s['unsafe_mean']:6.1f}"
            f" | maxV={s['max_v_buf_mean']:5.1f}"
            f" | maxC={s['max_c_uf_mean']:.4f}"
            f" | in_band={s['target_band_rate']:.1%}"
            f" | zero_unsafe={s['zero_unsafe_rate']:.1%}"
            f" | safe_target={s['safe_target_rate']:.1%}"
        )
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
