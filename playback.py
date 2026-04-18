"""
ESAC/TD3 模型回放脚本 - 生成完整回合动画 GIF
支持 9 维和 15 维状态模型
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from env.gym_env import ThickenerDewateringEnv
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algo", type=str, default="esac", choices=["esac", "td3"])
    parser.add_argument("--target", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", type=str, default="playback.gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def load_esac_state_dim(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a0 = ckpt.get('actor_0', None)
    if a0 is not None:
        return int(a0['fc1.weight'].shape[1])
    return 9


def main():
    args = parse_args()
    device = "cpu"
    checkpoint = Path(args.checkpoint).resolve()

    # 创建环境
    reward_config = RewardConfig(target_mass=args.target, max_steps=DEFAULT_CONTROL_STEPS)
    env = ThickenerDewateringEnv(
        max_steps=DEFAULT_CONTROL_STEPS,
        decision_interval=DEFAULT_DECISION_INTERVAL,
        target_mass=args.target,
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
    )

    # 从 checkpoint 读取 state_dim
    ckpt_state_dim = load_esac_state_dim(str(checkpoint), device)
    env_state_dim = env.observation_space.shape[0]
    needs_adapt = ckpt_state_dim != env_state_dim

    if needs_adapt:
        print(f"Checkpoint state_dim={ckpt_state_dim}, env state_dim={env_state_dim}, 使用前 {ckpt_state_dim} 维")

    # 加载模型
    if args.algo == "esac":
        from algorithms.esac import ESACAgent
        agent = ESACAgent(state_dim=ckpt_state_dim, action_dim=2, device=device)
    else:
        from algorithms.td3 import TD3Agent
        agent = TD3Agent(state_dim=ckpt_state_dim, action_dim=2, device=device)

    if not agent.load(str(checkpoint)):
        print("加载失败")
        return

    def select_action(state):
        s = state[:ckpt_state_dim] if needs_adapt else state
        return agent.select_action(s, deterministic=True)

    # ===== 运行一回合 =====
    print(f"Running episode with seed={args.seed}...")
    state, _ = env.reset(seed=args.seed)

    records = []
    episode_reward = 0.0

    # 初始记录
    records.append({
        'minute': 0, 'm_fp': 0.0, 'c_uf': float(state[0]),
        'v_buf': float(state[1]), 'q_uf': 0.0, 'q_fp': 0.0,
        'q_fp_cmd': 0.0,
        'qf': float(state[7]), 'cf': float(state[8]),
        'price': float(state[5]), 'energy': 0.0,
        'reward': 0.0, 'layers': extract_layers(env),
    })

    done = False
    while not done:
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        records.append({
            'minute': int(env.timecnt),
            'm_fp': float(info.get('current_mass', 0)),
            'c_uf': float(next_state[0]),
            'v_buf': float(next_state[1]),
            'q_uf': float(info.get('applied_q_uf', action[0])),
            'q_fp': float(info.get('applied_q_fp', action[1])),
            'q_fp_cmd': float(info.get('commanded_q_fp', action[1])),
            'qf': float(next_state[7]),
            'cf': float(next_state[8]),
            'price': float(next_state[5]),
            'energy': float(info.get('total_energy_cost', 0)),
            'reward': float(reward),
            'layers': extract_layers(env),
        })

        episode_reward += reward
        state = next_state

    final_mass = records[-1]['m_fp']
    print(f"Episode done. Final mass={final_mass:.1f}t, reward={episode_reward:.1f}")

    # ===== 创建动画 =====
    create_animation(records, checkpoint.name, args.target, args.output, args.fps, args.dpi)
    print(f"Saved: {args.output}")


def extract_layers(env):
    return [
        float(max(0.0, env.thickener.d2c(d / 1e6)))
        for d in env.thickener_state
    ]


def create_animation(records, run_name, target, output_path, fps, dpi):
    minutes = np.array([r['minute'] for r in records])
    mass = np.array([r['m_fp'] for r in records])
    c_uf = np.array([r['c_uf'] for r in records])
    v_buf = np.array([r['v_buf'] for r in records])
    q_uf = np.array([r['q_uf'] for r in records])
    q_fp = np.array([r['q_fp'] for r in records])
    qf = np.array([r['qf'] for r in records])
    cf = np.array([r['cf'] for r in records])
    price = np.array([r['price'] for r in records])
    energy = np.array([r['energy'] for r in records])
    layers = np.array([r['layers'] for r in records])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    ax_mass = fig.add_subplot(gs[0, 0])
    ax_cuf = fig.add_subplot(gs[0, 1])
    ax_vbuf = fig.add_subplot(gs[1, 0])
    ax_act = fig.add_subplot(gs[1, 1])
    ax_feed = fig.add_subplot(gs[2, 0])
    ax_price = fig.add_subplot(gs[2, 1])
    ax_tank = fig.add_subplot(gs[0:2, 2])
    ax_profile = fig.add_subplot(gs[2, 2])

    fig.suptitle(f"Episode Playback | {run_name} | seed=91", fontsize=13, fontweight='bold')

    x_max = max(float(minutes[-1]), 1.0)

    def setup(ax, title, ylabel):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Minute', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)

    setup(ax_mass, 'Cumulative Mass (t)', 'Mass')
    ax_mass.axhspan(target * 0.95, target * 1.1, color='#d8f3dc', alpha=0.6)
    ax_mass.axhline(target, color='#2d6a4f', ls='--', lw=1)
    ax_mass.set_ylim(0, max(mass.max() * 1.05, target * 1.3))

    setup(ax_cuf, 'Underflow Conc.', 'C_uf')
    ax_cuf.axhline(0.75, color='red', ls='--', lw=1)
    ax_cuf.set_ylim(0, max(0.85, c_uf.max() * 1.1))

    setup(ax_vbuf, 'Buffer Vol. (m³)', 'V_buf')
    ax_vbuf.axhline(30.0, color='red', ls='--', lw=1)
    ax_vbuf.set_ylim(0, max(33, v_buf.max() * 1.1))

    setup(ax_act, 'Actions', 'Flow (m³/h)')
    ax_act.set_ylim(0, 75)

    setup(ax_feed, 'Feed', 'Qf')
    ax_feed.set_ylim(34, 51)
    ax_feed2 = ax_feed.twinx()
    ax_feed2.set_ylabel('Cf', fontsize=7)
    ax_feed2.set_ylim(0.28, 0.47)

    setup(ax_price, 'Price & Energy', 'Price')
    ax_price.set_ylim(0.35, 1.3)
    ax_price2 = ax_price.twinx()
    ax_price2.set_ylabel('Energy', fontsize=7)
    ax_price2.set_ylim(0, max(1, energy.max() * 1.1))

    ax_tank.set_title('Thickener Layers', fontsize=9)
    ax_tank.set_xlim(-0.6, 0.6)
    ax_tank.set_ylim(0, 10)
    ax_tank.set_xticks([])
    ax_tank.set_yticks(np.arange(0.5, 10.5))
    ax_tank.set_yticklabels([f'L{10-i}' for i in range(10)], fontsize=6)

    ax_profile.set_title('Conc. Profile', fontsize=9)
    ax_profile.set_xlabel('Conc.', fontsize=7)
    ax_profile.set_ylabel('Layer', fontsize=7)
    ax_profile.set_xlim(0, max(0.85, layers.max() * 1.1))
    ax_profile.set_ylim(0.5, 10.5)
    ax_profile.set_yticks(range(1, 11))
    ax_profile.grid(True, alpha=0.25)

    # 绘制线
    m_line, = ax_mass.plot([], [], color='#1f77b4', lw=2)
    m_dot, = ax_mass.plot([], [], 'o', color='#0b3d91', ms=4)
    cuf_line, = ax_cuf.plot([], [], color='#6a4c93', lw=2)
    cuf_dot, = ax_cuf.plot([], [], 'o', color='#3c096c', ms=4)
    v_line, = ax_vbuf.plot([], [], color='#2a9d8f', lw=2)
    v_dot, = ax_vbuf.plot([], [], 'o', color='#1d7874', ms=4)
    quf_line, = ax_act.plot([], [], color='#f4a261', lw=2, label='Q_uf')
    qfp_line, = ax_act.plot([], [], color='#e63946', lw=2, label='Q_fp')
    qf_line, = ax_feed.plot([], [], color='#457b9d', lw=2, label='Qf')
    cf_line, = ax_feed2.plot([], [], color='#1d3557', lw=2, label='Cf')
    p_line, = ax_price.plot([], [], color='#ff006e', lw=2, label='Price')
    e_line, = ax_price2.plot([], [], color='#fb8500', lw=2, label='Energy')

    cursors = [ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.4)
               for ax in [ax_mass, ax_cuf, ax_vbuf, ax_act, ax_feed, ax_price]]

    ax_act.legend(loc='upper left', fontsize=6)
    ax_feed.legend(loc='upper left', fontsize=6)
    ax_price.legend(loc='upper left', fontsize=6)

    # 浓密机罐体
    cmap = plt.get_cmap('YlOrBr')
    rects = []
    txts = []
    for i in range(10):
        rect = plt.Rectangle((-0.35, i), 0.7, 1, facecolor=cmap(0), edgecolor='#555', lw=0.5)
        ax_tank.add_patch(rect)
        rects.append(rect)
        txt = ax_tank.text(0, i + 0.5, '', ha='center', va='center', fontsize=6)
        txts.append(txt)

    prof_line, = ax_profile.plot([], [], color='#8d0801', lw=2, marker='o', ms=3)
    ax_profile.axvline(0.75, color='red', ls='--', lw=0.8)

    info = fig.text(0.01, 0.02, '', fontsize=7, family='monospace', va='bottom')

    def update(frame):
        x = minutes[:frame+1]
        m_line.set_data(x, mass[:frame+1])
        m_dot.set_data([minutes[frame]], [mass[frame]])
        cuf_line.set_data(x, c_uf[:frame+1])
        cuf_dot.set_data([minutes[frame]], [c_uf[frame]])
        v_line.set_data(x, v_buf[:frame+1])
        v_dot.set_data([minutes[frame]], [v_buf[frame]])
        quf_line.set_data(x, q_uf[:frame+1])
        qfp_line.set_data(x, q_fp[:frame+1])
        qf_line.set_data(x, qf[:frame+1])
        cf_line.set_data(x, cf[:frame+1])
        p_line.set_data(x, price[:frame+1])
        e_line.set_data(x, energy[:frame+1])

        lv = layers[frame]
        for j in range(10):
            si = 9 - j
            val = float(lv[si])
            rects[j].set_facecolor(cmap(np.clip(val / 0.75, 0, 1)))
            txts[j].set_text(f'{val:.3f}')

        prof_line.set_data(lv, np.arange(1, 11))
        for c in cursors:
            c.set_xdata([minutes[frame], minutes[frame]])

        info.set_text(
            f"step={frame:3d}  min={records[frame]['minute']:4d}  "
            f"mass={records[frame]['m_fp']:.1f}t  C_uf={records[frame]['c_uf']:.4f}  "
            f"V_buf={records[frame]['v_buf']:.1f}\n"
            f"Q_uf={records[frame]['q_uf']:.1f}  Q_fp={records[frame]['q_fp']:.1f}  "
            f"Qf={records[frame]['qf']:.1f}  Cf={records[frame]['cf']:.3f}  "
            f"price={records[frame]['price']:.3f}  energy={records[frame]['energy']:.1f}"
        )

    anim = animation.FuncAnimation(fig, update, frames=len(records), interval=max(1, int(1000/fps)), blit=False)
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)


if __name__ == '__main__':
    main()
