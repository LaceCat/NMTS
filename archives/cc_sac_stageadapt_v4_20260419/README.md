# CC SAC Stage-Adapt v4

来源：
- 基座模型：`archives/cc_sac_ft2_20260418/checkpoint/best_model.pth`
- 微调运行：`runs/sac_cc_ft2_stageadapt_v4_crossonly_warm1500_probe12`

训练思路：
- 分段微调
- `warmup_steps=1500`
- 保持原始安全/经济奖励骨架
- 仅逐步增强“过 400t”激励，不额外叠加强浓度奖励

当前归档文件：
- `best_model.pth`
- `run_config.json`

5 种子固定评估（seed = 91~95）：

| 模型 | 平均产量 t | 平均能耗 | UnsafeEp | DryStep | LowCStep | 平均底流浓度 |
|---|---:|---:|---:|---:|---:|---:|
| 原始归档 baseline | 399.744 | 537.104 | 0.0 | 0.0 | 0.01 | 0.7137 |
| stage-adapt v4 | 400.435 | 550.316 | 0.0 | 0.0 | 0.01 | 0.7087 |

结论：
- 该版本已经稳定跨过 `400t`
- 全 5 种子 `unsafe=0`
- 干抽率维持为 `0`
- 代价是能耗较 baseline 增加约 `13.2`

建议用途：
- 作为“稳定达标版”CC 连续动作模型存档
- 后续若继续压低能耗，可从该模型继续做小步长微调
