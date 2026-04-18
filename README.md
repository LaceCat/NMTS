# 浓密脱水强化学习项目

当前仓库已经做过一轮清理，保留的是还在使用的源码、诊断脚本和正式训练结果。

## 当前重点

- 主研究对象：`CC` 动作空间
- 当前主算法：`SAC`
- 训练入口：`train.py`
- 评估入口：`evaluate.py`
- 正式训练结果统一放在 `runs/`

## 目录

```text
claude-code/
├─ algorithms/                 # 算法实现
├─ env/                        # 环境、物理模型、奖励
├─ utils/                      # buffer / metrics / visualization
├─ diagnostics/                # 诊断脚本与分析输出
├─ runs/                       # 正式训练结果
├─ train.py                    # 训练入口
├─ evaluate.py                 # 评估入口
├─ tune_sac_cc.py              # SAC(CC) 调参脚本
├─ CURRENT_MODEL_SELECTION.md  # 当前模型选择说明
└─ selected_models.json        # 当前选定模型
```

## 环境

```powershell
& 'F:\毕设\thickener1.12codex - 副本\thickener_rl_improved1.11.0\.venv\Scripts\Activate.ps1'
Set-Location 'F:\毕设\claude-code'
```

## 推荐训练命令

当前推荐从标准 `SAC + CC` 基线开始：

```powershell
python train.py `
  --algo sac `
  --mode CC `
  --device cuda `
  --epochs 20 `
  --episodes_per_epoch 2 `
  --eval_episodes 1 `
  --run_name sac_cc_cleanbaseline_v2 `
  --disable_post_target_fp_governor `
  --warmup_steps 300 `
  --uf_delta_max 3.0 `
  --q_fp_delta_max 12 `
  --sac_mean_q_weight 0.0 `
  --sac_std_reg_weight 0.0
```

## 推荐评估命令

```powershell
python evaluate.py `
  --checkpoint 'F:\毕设\claude-code\runs\sac_cc_cleanbaseline_v2\checkpoints\best_model.pth' `
  --algo sac `
  --mode CC `
  --device cuda `
  --uf_control_mode delta `
  --uf_delta_max 3.0 `
  --q_fp_delta_max 12 `
  --disable_post_target_fp_governor `
  --seeds 5 `
  --seed_start 91 `
  --verbose
```

## 当前结果摘要

- `sac_cc_cleanbaseline_v1`
  - 更稳，5 种子全部 `>= 400`
  - 能耗更高
- `sac_cc_cleanbaseline_v2`
  - 更省能耗
  - 安全性干净
  - 但部分种子略低于 `400`
- `sac_cc_cleanbaseline_v3_actorprior`
  - 已验证不如 `v2`
  - 不建议继续作为主线

更详细的模型选择说明见：

- `CURRENT_MODEL_SELECTION.md`
- `selected_models.json`

## 说明

- 根目录里的旧调试目录、一次性 checkpoint、缓存和临时图像已经清理。
- 如果后续还需要更激进地精简，可以继续删除旧 baseline 脚本和一次性分析脚本，但这次先保留源码层的可复用工具。
