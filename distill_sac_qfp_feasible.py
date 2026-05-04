"""
Offline/iterative Q_fp feasibility distillation for the single-layer SAC CC policy.

The old low-energy CC policy reaches the 400 t target, but its filter-press
command is often larger than the slurry inventory can physically support. The
environment clips the actual flow, which creates many dry-run flags. This tool
keeps the learned Q_uf rhythm and trains only the SAC actor's Q_fp mean head so
the policy itself emits the feasible/actual Q_fp seen during rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.sac import SACAgent
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from evaluate import _transform_obs_for_sac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill SAC Q_fp to feasible actual Q_fp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS)
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL)
    parser.add_argument("--uf_delta_max", type=float, default=3.0)
    parser.add_argument("--q_fp_delta_max", type=float, default=12.0)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--collect_seeds", type=str, default="91,92,93,94,95")
    parser.add_argument("--eval_seeds", type=str, default="91,92,93,94,95")
    parser.add_argument("--train_epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--qfp_weight", type=float, default=1.0)
    parser.add_argument("--qfp_bias_weight", type=float, default=0.0)
    parser.add_argument("--qfp_target_offset", type=float, default=0.0)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--q_uf_anchor_weight", type=float, default=2.0)
    parser.add_argument("--direct_q_fp_physical_only", action="store_true")
    parser.add_argument("--disable_soft_governors", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def parse_seed_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_env(args: argparse.Namespace) -> ThickenerDewateringEnv:
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        mode="CC",
        uf_control_mode="delta",
        uf_delta_max=args.uf_delta_max,
        q_fp_delta_max=None if args.q_fp_delta_max < 0 else args.q_fp_delta_max,
        pricing=PricingPresets.daily_24h(),
        reward_config=RewardConfig(target_mass=args.target, max_steps=args.steps),
        enable_post_target_fp_governor=not args.disable_soft_governors,
        enable_post_target_idle_seeker=not args.disable_soft_governors,
        enable_midcourse_quality_governor=not args.disable_soft_governors,
        enable_late_concentration_keeper=not args.disable_soft_governors,
        enable_late_target_compensator=not args.disable_soft_governors,
        direct_q_fp_physical_only=args.direct_q_fp_physical_only,
    )
    return env


def build_agent(args: argparse.Namespace, checkpoint: dict[str, Any]) -> SACAgent:
    action_low = np.asarray(checkpoint.get("action_low", [-args.uf_delta_max, 0.0]), dtype=np.float32)
    action_high = np.asarray(checkpoint.get("action_high", [args.uf_delta_max, 70.0]), dtype=np.float32)
    agent = SACAgent(
        state_dim=int(checkpoint.get("state_dim", 46)),
        action_dim=2,
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        action_low=action_low,
        action_high=action_high,
        use_gru_encoder=bool(checkpoint.get("use_gru_encoder", False)),
        gru_hidden_dim=int(checkpoint.get("gru_hidden_dim", 96)),
        q_uf_is_delta=True,
        device=args.device,
    )
    agent.load(args.checkpoint)
    return agent


def collect_dataset(agent: SACAgent, args: argparse.Namespace, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    states: list[np.ndarray] = []
    qfp_targets: list[float] = []
    q_uf_targets: list[float] = []
    dry_steps = 0
    unsafe_steps = 0
    masses = []
    energies = []

    for seed in seeds:
        env = make_env(args)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            state = _transform_obs_for_sac(obs, args.target, args.steps)
            action = agent.select_action(state, deterministic=True)
            next_obs, _, terminated, truncated, info = env.step(action)
            states.append(state.astype(np.float32))
            q_uf_targets.append(float(action[0]))
            qfp_targets.append(float(np.clip(float(info.get("actual_q_fp", 0.0)) + args.qfp_target_offset, 0.0, 70.0)))
            dry_steps += int(bool(info.get("dry_run_violation", False)))
            unsafe_steps += int(bool(info.get("safety_violation", False)))
            obs = next_obs
            done = bool(terminated or truncated)
        masses.append(float(info.get("current_mass", 0.0)))
        energies.append(float(info.get("total_energy_cost", 0.0)))

    metrics = {
        "mass": float(np.mean(masses)) if masses else 0.0,
        "energy": float(np.mean(energies)) if energies else 0.0,
        "dry_rate": float(dry_steps / max(len(seeds) * args.steps, 1)),
        "unsafe_rate": float(unsafe_steps / max(len(seeds) * args.steps, 1)),
    }
    return (
        np.stack(states),
        np.asarray(qfp_targets, dtype=np.float32),
        np.asarray(q_uf_targets, dtype=np.float32),
        metrics,
    )


def train_qfp_head(
    agent: SACAgent,
    states: np.ndarray,
    qfp_targets: np.ndarray,
    q_uf_targets: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    for param in agent.actor.parameters():
        param.requires_grad_(bool(args.train_backbone))
    if not hasattr(agent.actor, "mean_head"):
        raise RuntimeError("This script expects a SAC actor with mean_head.")
    for param in agent.actor.mean_head.parameters():
        param.requires_grad_(True)
    if hasattr(agent.actor, "log_std_head"):
        for param in agent.actor.log_std_head.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in agent.actor.parameters() if p.requires_grad], lr=args.lr)
    x = torch.tensor(states, dtype=torch.float32, device=args.device)
    y_qfp = torch.tensor(qfp_targets, dtype=torch.float32, device=args.device).view(-1)
    y_quf = torch.tensor(q_uf_targets, dtype=torch.float32, device=args.device).view(-1)
    n = x.shape[0]
    last_loss = 0.0
    last_mae = 0.0

    for _ in range(max(args.train_epochs, 1)):
        perm = torch.randperm(n, device=args.device)
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch_x = x[idx]
            batch_y = y_qfp[idx]
            mean, _ = agent.actor(batch_x)
            det = torch.tanh(mean) * agent.action_scale + agent.action_bias
            qfp = det[:, 1]
            q_uf = det[:, 0]
            loss = args.qfp_weight * F.mse_loss(qfp / 70.0, batch_y / 70.0)
            if args.train_backbone and args.q_uf_anchor_weight > 0.0:
                loss = loss + args.q_uf_anchor_weight * F.mse_loss(q_uf / max(args.uf_delta_max, 1e-6), y_quf[idx] / max(args.uf_delta_max, 1e-6))
            if args.qfp_bias_weight > 0.0:
                # Mild asymmetric term: over-commanding is what creates dry-run.
                loss = loss + args.qfp_bias_weight * torch.relu(qfp - batch_y).pow(2).mean() / (70.0 ** 2)
            optimizer.zero_grad()
            loss.backward()
            if not args.train_backbone:
                # Keep Q_uf row exactly anchored. Only the Q_fp row is allowed to move.
                if agent.actor.mean_head.weight.grad is not None:
                    agent.actor.mean_head.weight.grad[0].zero_()
                if agent.actor.mean_head.bias.grad is not None:
                    agent.actor.mean_head.bias.grad[0].zero_()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        mean, _ = agent.actor(x)
        det = torch.tanh(mean) * agent.action_scale + agent.action_bias
        last_mae = float(torch.mean(torch.abs(det[:, 1] - y_qfp)).cpu().item())
    return {"loss": last_loss, "qfp_mae": last_mae}


def evaluate_agent(agent: SACAgent, args: argparse.Namespace, seeds: list[int]) -> dict[str, float]:
    masses = []
    energies = []
    avg_cufs = []
    dry_steps = 0
    unsafe_steps = 0
    low_conc_steps = 0
    total_steps = 0
    qfp_gaps = []
    for seed in seeds:
        env = make_env(args)
        obs, _ = env.reset(seed=seed)
        done = False
        cufs = []
        while not done:
            state = _transform_obs_for_sac(obs, args.target, args.steps)
            action = agent.select_action(state, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_steps += 1
            dry_steps += int(bool(info.get("dry_run_violation", False)))
            unsafe_steps += int(bool(info.get("safety_violation", False)))
            low_conc_steps += int(bool(info.get("low_conc_violation", False)))
            qfp_gaps.append(max(float(info.get("applied_q_fp", 0.0)) - float(info.get("actual_q_fp", 0.0)), 0.0))
            cufs.append(float(info.get("c_uf", 0.0)))
        masses.append(float(info.get("current_mass", 0.0)))
        energies.append(float(info.get("total_energy_cost", 0.0)))
        avg_cufs.append(float(np.mean(cufs)) if cufs else 0.0)
    return {
        "mean_final_mass": float(np.mean(masses)),
        "std_final_mass": float(np.std(masses)),
        "mean_final_energy": float(np.mean(energies)),
        "std_final_energy": float(np.std(energies)),
        "mean_c_uf": float(np.mean(avg_cufs)),
        "dry_run_step_rate": float(dry_steps / max(total_steps, 1)),
        "unsafe_step_rate": float(unsafe_steps / max(total_steps, 1)),
        "low_conc_step_rate": float(low_conc_steps / max(total_steps, 1)),
        "mean_qfp_actual_gap": float(np.mean(qfp_gaps)) if qfp_gaps else 0.0,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    agent = build_agent(args, checkpoint)
    collect_seeds = parse_seed_list(args.collect_seeds)
    eval_seeds = parse_seed_list(args.eval_seeds)
    history: list[dict[str, Any]] = []
    all_states = []
    all_qfp_targets = []
    all_q_uf_targets = []

    initial_eval = evaluate_agent(agent, args, eval_seeds)
    print("initial", json.dumps(initial_eval, ensure_ascii=False, indent=2))
    history.append({"round": 0, "eval": initial_eval})

    for round_idx in range(1, args.rounds + 1):
        states, qfp_targets, q_uf_targets, collect_metrics = collect_dataset(agent, args, collect_seeds)
        all_states.append(states)
        all_qfp_targets.append(qfp_targets)
        all_q_uf_targets.append(q_uf_targets)
        train_states = np.concatenate(all_states, axis=0)
        train_qfp_targets = np.concatenate(all_qfp_targets, axis=0)
        train_q_uf_targets = np.concatenate(all_q_uf_targets, axis=0)
        train_metrics = train_qfp_head(agent, train_states, train_qfp_targets, train_q_uf_targets, args)
        eval_metrics = evaluate_agent(agent, args, eval_seeds)
        record = {
            "round": round_idx,
            "collect": collect_metrics,
            "train": train_metrics,
            "eval": eval_metrics,
            "dataset_size": int(train_states.shape[0]),
        }
        history.append(record)
        print("round", round_idx, json.dumps(record, ensure_ascii=False, indent=2))
        agent.save(str(out_dir / "checkpoints" / f"round_{round_idx}.pth"))

    final_path = out_dir / "checkpoints" / "final_model.pth"
    agent.save(str(final_path))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "final_model": str(final_path.resolve()),
                "args": vars(args),
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
