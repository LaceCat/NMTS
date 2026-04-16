"""
Evaluation entry for the thickener dewatering RL experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.td3 import TD3Agent
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.gym_env import ThickenerDewateringEnv
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from utils.metrics import compute_pass_rate, evaluate_episode
from utils.visualization import save_episode_plot


def parse_args():
    parser = argparse.ArgumentParser(description="RL evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--algo", type=str, default="esac", choices=["esac", "td3"], help="Algorithm")
    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps per episode")
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_DECISION_INTERVAL,
        help="Physical minutes per decision step",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of evaluation seeds")
    parser.add_argument("--seed_start", type=int, default=91, help="Starting seed")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode details")
    parser.add_argument("--batch", action="store_true", help="Reserved batch mode flag")
    parser.add_argument("--save_plots", action="store_true", help="Save episode plots")
    return parser.parse_args()


def evaluate_single(agent, env, seed=0, verbose=False, save_plot=False, adapt_fn=None):
    state, _ = env.reset(seed=seed)
    done = False
    episode_reward = 0.0
    states_history = []
    actions_history = []
    rewards_history = []
    energy_history = []
    info_list = []

    while not done:
        if adapt_fn is not None:
            action = adapt_fn(state, deterministic=True)
        else:
            action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states_history.append(state.copy())
        actions_history.append(action.copy())
        rewards_history.append(reward)
        energy_history.append(info.get("total_energy_cost", 0.0))
        info_list.append(info)

        episode_reward += reward
        state = next_state

    if verbose:
        print(f"\nEpisode (seed={seed}):")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Final Mass: {info.get('current_mass', 0):.1f} t")
        print(f"  Energy Cost: {info.get('total_energy_cost', 0):.2f}")
        print(f"  Target Reached: {info.get('target_reached', False)}")

    if save_plot:
        os.makedirs("plots", exist_ok=True)
        save_episode_plot(
            np.array(states_history),
            np.array(actions_history),
            np.array(rewards_history),
            np.array(energy_history),
            save_path=f"plots/episode_seed{seed}.png",
        )

    metrics = evaluate_episode(
        info_list,
        env.reward_config.target_mass_low,
        env.reward_config.target_mass_high,
    )
    metrics["reward"] = episode_reward
    return metrics


def build_agent_and_adapter(args, env):
    if args.algo == "esac":
        if args.mode != "CC":
            raise ValueError("ESAC evaluation currently only supports true CC mode.")

        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        first_actor_key = checkpoint.get("actor_0", None)
        if first_actor_key is not None:
            ckpt_state_dim = first_actor_key["fc1.weight"].shape[1]
        else:
            ckpt_state_dim = env.observation_space.shape[0]

        from algorithms.esac import ESACAgent

        agent = ESACAgent(
            state_dim=ckpt_state_dim,
            action_dim=env.action_space.shape[0],
            device=args.device,
        )

        env_state_dim = env.observation_space.shape[0]
        needs_adapt = ckpt_state_dim != env_state_dim
        if needs_adapt:
            print(f"Warning: checkpoint state_dim={ckpt_state_dim}, env state_dim={env_state_dim}")
            print(f"  Using first {ckpt_state_dim} dimensions of observation for evaluation.")

        def select_action_adapted(state, deterministic=True):
            return agent.select_action(state[:ckpt_state_dim], deterministic=deterministic)

        return agent, needs_adapt, select_action_adapted

    if args.mode == "DD":
        from algorithms.discrete_ddqn import DiscreteDDQNAgent

        agent = DiscreteDDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
        return agent, False, None

    if args.mode == "CD":
        from algorithms.hybrid_td3 import HybridTD3Agent

        agent = HybridTD3Agent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
        return agent, False, None

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_state_dim = env.observation_space.shape[0]
    actor_key = checkpoint.get("actor", None)
    if actor_key is not None and "net.0.weight" in actor_key:
        ckpt_state_dim = actor_key["net.0.weight"].shape[1]

    agent = TD3Agent(
        state_dim=ckpt_state_dim,
        action_dim=env.action_space.shape[0],
        device=args.device,
    )

    env_state_dim = env.observation_space.shape[0]
    needs_adapt = ckpt_state_dim != env_state_dim
    if needs_adapt:
        print(f"Warning: checkpoint state_dim={ckpt_state_dim}, env state_dim={env_state_dim}")
        print(f"  Using first {ckpt_state_dim} dimensions of observation for evaluation.")

    def select_action_adapted(state, deterministic=True):
        return agent.select_action(state[:ckpt_state_dim], deterministic=deterministic)

    return agent, needs_adapt, select_action_adapted


def main():
    args = parse_args()

    reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        mode=args.mode,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
    )

    agent, needs_adapt, select_action_adapted = build_agent_and_adapter(args, env)
    if not agent.load(args.checkpoint):
        print("Failed to load checkpoint.")
        return

    print(f"\nEvaluating: {args.checkpoint} (algo={args.algo})")
    print(
        f"Target={args.target} t | mode={args.mode} | steps={args.steps} | "
        f"seeds={args.seed_start}~{args.seed_start + args.seeds - 1}"
    )

    all_metrics = []
    for i in range(args.seeds):
        seed = args.seed_start + i
        metrics = evaluate_single(
            agent,
            env,
            seed=seed,
            verbose=args.verbose,
            save_plot=args.save_plots or (i < 2),
            adapt_fn=select_action_adapted if needs_adapt else None,
        )
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        return

    print("\n" + "=" * 50)
    print("Batch evaluation:")
    print(f"  valid samples: {len(all_metrics)}/{args.seeds}")

    final_masses = [m["final_mass"] for m in all_metrics]
    energy_costs = [m["energy_cost"] for m in all_metrics]
    rewards = [m["reward"] for m in all_metrics]
    eeis = [m["eei"] for m in all_metrics]

    print(f"  final mass:   {np.mean(final_masses):.1f} +/- {np.std(final_masses):.1f} t")
    print(f"  energy cost:  {np.mean(energy_costs):.2f} +/- {np.std(energy_costs):.2f}")
    print(f"  reward:       {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  EEI:          {np.mean(eeis):.2f} +/- {np.std(eeis):.2f}")

    target_low = reward_config.target_mass_low
    target_high = reward_config.target_mass_high
    pass_rate = compute_pass_rate(final_masses, target_low, target_high)
    print(f"  in-band rate: {pass_rate:.1%}")

    os.makedirs("logs", exist_ok=True)
    eval_result = {
        "checkpoint": args.checkpoint,
        "target": args.target,
        "mode": args.mode,
        "steps": args.steps,
        "seeds": f"{args.seed_start}~{args.seed_start + args.seeds - 1}",
        "n_samples": len(all_metrics),
        "mean_mass": float(np.mean(final_masses)),
        "std_mass": float(np.std(final_masses)),
        "mean_energy": float(np.mean(energy_costs)),
        "mean_reward": float(np.mean(rewards)),
        "mean_eei": float(np.mean(eeis)),
        "pass_rate": float(pass_rate),
        "per_seed": [
            {
                "seed": args.seed_start + i,
                "mass": m["final_mass"],
                "energy": m["energy_cost"],
                "reward": m["reward"],
                "eei": m["eei"],
            }
            for i, m in enumerate(all_metrics)
        ],
    }

    eval_path = "logs/eval_result.json"
    with open(eval_path, "w", encoding="utf-8") as file:
        json.dump(eval_result, file, indent=2)
    print(f"\nSaved evaluation summary to: {eval_path}")


if __name__ == "__main__":
    main()
