# 浓密脱水过程强化学习协同优化

基于 TD3 的湿法冶金浓密脱水过程双连续控制模型。

## 工艺流程

```
进料 (Qf, Cf) → 浓密机 → 底流泵 (Q_uf) → 搅拌罐 → 压滤泵 (Q_fp) → 压滤机
```

## 项目结构

```
├── env/                    # 环境模块
│   ├── physics/            # 物理模型
│   │   ├── thickener.py    # Bürger 浓密机模型
│   │   └── buffer_press.py # 搅拌罐+压滤模型
│   ├── reward/             # 奖励函数
│   │   ├── config.py       # 奖励配置
│   │   ├── scheme.py       # 奖励计算
│   │   ├── schemes.py      # 奖励方案定义
│   │   └── pricing.py      # 分时电价
│   └── gym_env.py          # Gymnasium 环境
├── algorithms/             # RL 算法
│   └── td3.py             # TD3 实现
├── utils/                  # 工具
│   ├── replay_buffer.py    # 优先经验回放
│   ├── networks.py         # 网络定义
│   ├── metrics.py          # 指标计算
│   └── visualization.py    # 可视化
├── train.py                # 训练入口
└── evaluate.py             # 评估入口
```

## 快速开始

### 环境

```bash
# 使用已有的虚拟环境
& 'F:\毕设\thickener1.12codex - 副本\thickener_rl_improved1.11.0\.venv\Scripts\Activate.ps1'
```

### 训练

```bash
# 默认配置 (400t, 288步, 24h)
python train.py

# 自定义参数
python train.py --epochs 500 --target 400 --steps 288 --interval 5

# 从检查点继续训练
python train.py --checkpoint checkpoints/best_model.pth
```

### 评估

```bash
# 单回合评估
python evaluate.py --checkpoint checkpoints/best_model.pth --verbose

# 批量评估 (10 个种子)
python evaluate.py --checkpoint checkpoints/best_model.pth --seeds 10 --seed_start 91 --batch
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| target_mass | 400 | 目标干矿量 (t) |
| max_steps | 288 | 每回合策略步数 |
| decision_interval | 5 | 每策略步物理分钟数 |
| Q_uf 范围 | [0, 50] | 底流泵流量 (m³/h) |
| Q_fp 范围 | [0, 70] | 压滤泵流量 (m³/h) |
| 达标范围 | [380, 440] | 目标 95%-110% |

## 修正的问题

1. **泵名对调**: 原始代码 action[0]/action[1] 与 Q_uf/Q_fp 对应关系错误
2. **奖励函数**: safety_penalty 符号错误、产出奖励被注释、目标范围不匹配
3. **算法不匹配**: 原始使用 Hybrid SAC (混合动作)，论文要求 TD3 (双连续)
