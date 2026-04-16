"""
可视化工具

训练曲线、回合轨迹图
"""

import os
import numpy as np
from typing import Optional


def save_training_plot(
    rewards: list,
    final_masses: list,
    energy_costs: list,
    save_dir: str = "plots",
):
    """保存训练过程可视化"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Process', fontsize=16)

    # 奖励
    if rewards:
        axs[0, 0].plot(rewards, 'b-')
    axs[0, 0].set_title('Episode Reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)

    # 干矿量
    if final_masses:
        axs[0, 1].plot(final_masses, 'g-')
    axs[0, 1].set_title('Final Dry Mass')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Mass (t)')
    axs[0, 1].grid(True)

    # 能耗
    if energy_costs:
        axs[1, 0].plot(energy_costs, 'r-')
    axs[1, 0].set_title('Total Energy Cost')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Cost')
    axs[1, 0].grid(True)

    axs[1, 1].axis('off')
    axs[1, 1].text(0.5, 0.5, 'Training Complete', ha='center', va='center', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    plt.close()
    print(f"训练曲线已保存到 {os.path.join(save_dir, 'training_metrics.png')}")


def save_episode_plot(
    states_history: np.ndarray,
    actions_history: np.ndarray,
    rewards_history: np.ndarray,
    energy_history: np.ndarray,
    save_path: str = "plots/episode.png",
):
    """保存单回合轨迹图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Episode Trajectory', fontsize=16)

    steps = np.arange(len(states_history))

    # 状态
    axs[0, 0].plot(steps, states_history[:, 3], 'b-')  # M_FP
    axs[0, 0].set_title('Cumulative Dry Mass')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Mass (t)')
    axs[0, 0].grid(True)

    axs[0, 1].plot(steps, states_history[:, 1], 'g-')  # V_buf
    axs[0, 1].set_title('Buffer Volume')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Volume (m³)')
    axs[0, 1].grid(True)

    # 动作
    axs[1, 0].plot(steps, actions_history[:, 0], 'orange')
    axs[1, 0].set_title('Underflow Pump (Q_uf)')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Flow (m³/h)')
    axs[1, 0].grid(True)

    axs[1, 1].plot(steps, actions_history[:, 1], 'red')
    axs[1, 1].set_title('Filter Press Pump (Q_fp)')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Flow (m³/h)')
    axs[1, 1].grid(True)

    # 奖励和能耗
    axs[2, 0].plot(steps, rewards_history, 'purple')
    axs[2, 0].set_title('Step Reward')
    axs[2, 0].set_xlabel('Step')
    axs[2, 0].set_ylabel('Reward')
    axs[2, 0].grid(True)

    axs[2, 1].plot(steps, energy_history, 'brown')
    axs[2, 1].set_title('Cumulative Energy Cost')
    axs[2, 1].set_xlabel('Step')
    axs[2, 1].set_ylabel('Cost')
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"回合轨迹已保存到 {save_path}")
