"""
Evaluation entry for the thickener dewatering RL experiments.
"""

import argparse
import json
import os
import sys

import numpy as np

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


def evaluate_single(agent, env, seed=0, verbose=False, save_plot=False):
    state, _ = env.reset(seed=seed)
    done = False
    episode_reward = 0.0
    states_history = []
    actions_history = []
    rewards_history = []
    energy_history = []
    info_list = []

    while not done:
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


def main():
    args = parse_args()

    reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
    )

    if args.algo == "esac":
        from algorithms.esac import ESACAgent

        agent = ESACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
    else:
        agent = TD3Agent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=args.device,
        )

    if not agent.load(args.checkpoint):
        print("Failed to load checkpoint.")
        return

    print(f"\nEvaluating: {args.checkpoint} (algo={args.algo})")
    print(
        f"Target={args.target} t | steps={args.steps} | "
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
