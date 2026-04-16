"""
TD3 评估入口

用法:
    python evaluate.py --checkpoint checkpoints/best_model.pth        # 单回合评估
    python evaluate.py --checkpoint checkpoints/best_model.pth --batch  # 批量评估
    python evaluate.py --checkpoint checkpoints/best_model.pth --seeds 10  # 10 个种子
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.gym_env import ThickenerDewateringEnv
from env.reward.pricing import PricingPresets
from env.reward.config import RewardConfig
from algorithms.td3 import TD3Agent
from utils.metrics import compute_pass_rate, evaluate_episode
from utils.visualization import save_episode_plot


def parse_args():
    parser = argparse.ArgumentParser(description='RL 评估')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--algo', type=str, default='esac', choices=['esac', 'td3'],
                        help='算法 (需与训练时一致)')
    parser.add_argument('--target', type=float, default=400.0, help='目标干矿量')
    parser.add_argument('--steps', type=int, default=288, help='每回合步数')
    parser.add_argument('--interval', type=int, default=5, help='策略步物理分钟')
    parser.add_argument('--seeds', type=int, default=5, help='评估种子数')
    parser.add_argument('--seed_start', type=int, default=91, help='起始种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--verbose', action='store_true', help='详细信息')
    parser.add_argument('--batch', action='store_true', help='批量评估模式')
    parser.add_argument('--save_plots', action='store_true', help='保存轨迹图')
    return parser.parse_args()


def evaluate_single(agent, env, seed=0, verbose=False, save_plot=False):
    """单回合评估"""
    state, _ = env.reset(seed=seed)
    done = False
    episode_reward = 0
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
        energy_history.append(info.get('total_energy_cost', 0.0))
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
        os.makedirs('plots', exist_ok=True)
        save_episode_plot(
            np.array(states_history),
            np.array(actions_history),
            np.array(rewards_history),
            np.array(energy_history),
            save_path=f'plots/episode_seed{seed}.png',
        )

    metrics = evaluate_episode(
        info_list,
        env.reward_config.target_mass_low,
        env.reward_config.target_mass_high,
    )
    metrics['reward'] = episode_reward
    return metrics


def main():
    args = parse_args()

    # 初始化环境
    reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
    )

    # 加载模型
    if args.algo == 'esac':
        from algorithms.esac import ESACAgent
        agent = ESACAgent(device=args.device)
    else:
        from algorithms.td3 import TD3Agent
        agent = TD3Agent(device=args.device)

    if not agent.load(args.checkpoint):
        print("加载模型失败，退出")
        return

    print(f"\n评估模型: {args.checkpoint} (算法: {args.algo})")
    print(f"目标: {args.target} t, 步数: {args.steps}, 种子: {args.seed_start}~{args.seed_start + args.seeds - 1}")

    all_metrics = []
    for i in range(args.seeds):
        seed = args.seed_start + i
        metrics = evaluate_single(
            agent, env, seed=seed,
            verbose=args.verbose,
            save_plot=args.save_plots or (i < 2),
        )
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print("\n" + "=" * 50)
        print("批量评估结果:")
        print(f"  有效样本: {len(all_metrics)}/{args.seeds}")

        final_masses = [m['final_mass'] for m in all_metrics]
        energy_costs = [m['energy_cost'] for m in all_metrics]
        rewards = [m['reward'] for m in all_metrics]
        eeis = [m['eei'] for m in all_metrics]

        print(f"  最终干矿量: {np.mean(final_masses):.1f} +/- {np.std(final_masses):.1f} t")
        print(f"  能耗成本:   {np.mean(energy_costs):.2f} +/- {np.std(energy_costs):.2f}")
        print(f"  平均奖励:   {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"  平均 EEI:  {np.mean(eeis):.2f} +/- {np.std(eeis):.2f}")

        target_low = reward_config.target_mass_low
        target_high = reward_config.target_mass_high
        pass_rate = compute_pass_rate(final_masses, target_low, target_high)
        print(f"  达标率:    {pass_rate:.1%}")

        # 保存评估结果
        os.makedirs('logs', exist_ok=True)
        eval_result = {
            'checkpoint': args.checkpoint,
            'target': args.target,
            'steps': args.steps,
            'seeds': f"{args.seed_start}~{args.seed_start + args.seeds - 1}",
            'n_samples': len(all_metrics),
            'mean_mass': float(np.mean(final_masses)),
            'std_mass': float(np.std(final_masses)),
            'mean_energy': float(np.mean(energy_costs)),
            'mean_reward': float(np.mean(rewards)),
            'mean_eei': float(np.mean(eeis)),
            'pass_rate': float(pass_rate),
            'per_seed': [
                {
                    'seed': args.seed_start + i,
                    'mass': m['final_mass'],
                    'energy': m['energy_cost'],
                    'reward': m['reward'],
                    'eei': m['eei'],
                }
                for i, m in enumerate(all_metrics)
            ],
        }
        eval_path = 'logs/eval_result.json'
        with open(eval_path, 'w') as f:
            json.dump(eval_result, f, indent=2)
        print(f"\n评估结果已保存到: {eval_path}")


if __name__ == '__main__':
    main()
